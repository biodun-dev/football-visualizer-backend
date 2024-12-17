import torch
from ultralytics import YOLO
from elements.deep_sort import DEEPSORT
from perspective_transform.position_mapping import get_mapped_position, transform_matrix
from arguments import Arguments
from team_assignment import TeamAssignment
import warnings
import os
import cv2
import csv
from shapely.geometry import box as shapely_box
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


# IOU calculation
def iou(bbox1, bbox2):
    box1 = shapely_box(*bbox1)
    box2 = shapely_box(*bbox2)
    intersection = box1.intersection(box2).area
    union = box1.union(box2).area
    return intersection / union if union > 0 else 0


# Consolidate detections
def consolidate_detections(detections, iou_threshold=0.9):
    consolidated = []
    for det in detections:
        bbox = det['bbox']
        label = det['label']
        score = det['score']
        found = False
        for con in consolidated:
            if con['label'] == label and abs(con['score'] - score) < 1e-3 and iou(bbox, con['bbox']) > iou_threshold:
                found = True
                break
        if not found:
            consolidated.append(det)
    return consolidated


# Resolve label conflicts
def resolve_label_conflicts(detections):
    resolved = []
    for det in detections:
        existing = next((d for d in resolved if iou(d['bbox'], det['bbox']) > 0.8), None)
        if existing:
            if existing['label'] == "player" and det['label'] == "referee":
                existing['label'] = "referee"
            elif existing['label'] == "referee" and det['label'] == "player":
                continue
        else:
            resolved.append(det)
    return resolved


# Assign teams to players based on position
def assign_team_to_players(player_positions):
    for player in player_positions:
        if player["position"][0] < 50:  # Assuming the pitch is 100 units wide
            player["team"] = "Team A"
        else:
            player["team"] = "Team B"
    return player_positions


# Infer last touch based on proximity
def infer_last_touch(ball_position, player_positions):
    if not player_positions or not ball_position:
        return None
    closest_player = min(
        player_positions,
        key=lambda player: np.linalg.norm(np.array(player["position"]) - np.array(ball_position))
    )
    return closest_player["team"]


# Event detection based on rules
def detect_event(ball_position, last_touch, pitch_bounds, ball_velocity, closest_player):
    x, y = ball_position
    event = None

    # Throw-in
    if y < pitch_bounds["top"] or y > pitch_bounds["bottom"]:
        event = "Throw-in"

    # Corner Kick or Goal Kick
    elif x < pitch_bounds["left"]:
        event = "Goal Kick" if last_touch == "Team A" else "Corner Kick"
    elif x > pitch_bounds["right"]:
        event = "Corner Kick" if last_touch == "Team B" else "Goal Kick"

    # Penalty
    elif pitch_bounds["penalty_left"] < x < pitch_bounds["penalty_right"] and \
            pitch_bounds["penalty_top"] < y < pitch_bounds["penalty_bottom"]:
        if closest_player and closest_player["team"] == "Team B":
            event = "Penalty"
        else:
            event = "Free-Kick"

    # Shot on Goal
    elif abs(x - pitch_bounds["right"]) < 5 and ball_velocity > 10:
        event = "Shot on Goal"

    # Dribble
    elif closest_player and ball_velocity < 2:
        event = "Dribble"

    # Pass
    elif ball_velocity > 5 and last_touch:
        event = "Pass"

    return event


# Main function
def main(opt):
    yolov8_model_path = 'weights/yolov8n.pt'
    detector = YOLO(yolov8_model_path)
    deep_sort = DEEPSORT(opt.deepsort_config)
    team_assignment_handler = TeamAssignment()
    homography_matrix = transform_matrix()

    confidence_thresholds = {
        "player": 0.8,
        "referee": 0.92,
        "goalkeeper": 0.9,
        "ball": 0.8
    }

    pitch_bounds = {
        "left": 0,
        "right": 100,
        "top": 0,
        "bottom": 50,
        "penalty_left": 10,
        "penalty_right": 90,
        "penalty_top": 10,
        "penalty_bottom": 40
    }

    yolo_csv_path = os.path.join(os.getcwd(), 'inference/output', 'yolo_detections.csv')
    events_csv_path = os.path.join(os.getcwd(), 'inference/output', 'events.csv')
    player_activity_csv_path = os.path.join(os.getcwd(), 'inference/output', 'player_activities.csv')

    os.makedirs(os.path.dirname(yolo_csv_path), exist_ok=True)

    yolo_headers = ['Frame', 'Object Label', 'Bounding Box', 'Confidence', 'Tracking ID', 'Pitch X', 'Pitch Y', 'Team']
    event_headers = ['Frame', 'Event', 'Ball X', 'Ball Y', 'Last Touch']
    player_activity_headers = ['Frame', 'Tracking ID', 'Team', 'Position X', 'Position Y', 'Activity']

    with open(yolo_csv_path, mode='w', newline='') as yolo_file, \
            open(events_csv_path, mode='w', newline='') as events_file, \
            open(player_activity_csv_path, mode='w', newline='') as player_activity_file:

        yolo_writer = csv.writer(yolo_file)
        event_writer = csv.writer(events_file)
        player_activity_writer = csv.writer(player_activity_file)

        yolo_writer.writerow(yolo_headers)
        event_writer.writerow(event_headers)
        player_activity_writer.writerow(player_activity_headers)

        cap = cv2.VideoCapture(opt.source)
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame.")
                break

            results = detector.predict(source=frame, conf=opt.conf_thresh)
            try:
                yolo_output = resolve_label_conflicts(consolidate_detections([
                    {'label': detector.names[int(box.cls[0])], 'bbox': box.xyxy[0].tolist(), 'score': box.conf[0].item()}
                    for box in results[0].boxes
                ]))
            except Exception as e:
                print(f"Error processing YOLO results: {e}")
                continue

            detections = []
            ball_position = None
            player_positions = []

            for obj in yolo_output:
                label = obj["label"]
                confidence = obj["score"]
                if confidence >= confidence_thresholds.get(label, 0.8):
                    detections.append(obj["bbox"] + [confidence])
                    mapped_position = get_mapped_position(obj["bbox"], homography_matrix)
                    if label == "ball":
                        ball_position = mapped_position
                    elif label == "player":
                        player_positions.append({"team": "unknown", "position": mapped_position})

            player_positions = assign_team_to_players(player_positions)
            last_touch = infer_last_touch(ball_position, player_positions)

            if detections:
                deep_sort.detection_to_deepsort(detections, frame)
                for track in deep_sort.deepsort.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    tracking_id = track.track_id
                    bbox = track.to_tlbr()
                    pitch_x, pitch_y = get_mapped_position(bbox, homography_matrix)

                    for obj in yolo_output:
                        label = obj['label']
                        confidence = obj['score']
                        if confidence >= confidence_thresholds.get(label, 0.8):
                            team_color = team_assignment_handler.assign_team_to_player(frame, bbox, label)
                            yolo_writer.writerow([frame_num, label, bbox, confidence, tracking_id, pitch_x, pitch_y, team_color])
                            activity = "Dribble" if ball_position and np.linalg.norm(np.array([pitch_x, pitch_y]) - np.array(ball_position)) < 2 else "Running"
                            player_activity_writer.writerow([frame_num, tracking_id, team_color, pitch_x, pitch_y, activity])

            if ball_position:
                ball_velocity = np.linalg.norm(np.array(ball_position))  # Placeholder for velocity computation
                closest_player = min(player_positions, key=lambda p: np.linalg.norm(np.array(p["position"]) - np.array(ball_position))) if player_positions else None
                event = detect_event(ball_position, last_touch, pitch_bounds, ball_velocity, closest_player)
                if event:
                    event_writer.writerow([frame_num, event, ball_position[0], ball_position[1], last_touch])

            # Overlay detections and labels
            for obj in yolo_output:
                label = obj['label']
                bbox = obj['bbox']  # [x1, y1, x2, y2]
                confidence = obj['score']

                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                color = (0, 255, 0)  # Green for default labels
                if label == "ball":
                    color = (0, 0, 255)  # Red for ball
                elif label == "referee":
                    color = (255, 0, 0)  # Blue for referee

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Add label text
                label_text = label
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Show the frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_num += 1

        cap.release()
        cv2.destroyAllWindows()


# Entry poin
if __name__ == '__main__':
    opt = Arguments().parse()
    with torch.no_grad():
        main(opt)
