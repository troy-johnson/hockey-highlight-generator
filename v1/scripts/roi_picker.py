import argparse
import cv2
import json
import sys

def first_frame(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Can't open {path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Can't read frame from {path}")
    return frame

def clamp_rect(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    x = min(x1, x2)
    y = min(y1, y2)
    rw = abs(x2 - x1)
    rh = abs(y2 - y1)
    return x, y, rw, rh

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam1")
    ap.add_argument("cam2")
    ap.add_argument("--out", required=True, help="Path to write rois.json")
    args = ap.parse_args()

    cam_paths = [args.cam1, args.cam2]
    out_path = args.out

    rois = {
        "camera_1": {"net": None, "slot": None},
        "camera_2": {"net": None, "slot": None},
    }

    frames = [first_frame(cam_paths[0]), first_frame(cam_paths[1])]
    cam_idx = 0

    # Drawing state
    drawing = False
    start_pt = None
    current_rect = None  # (x,y,w,h) live while drawing
    last_rect = None     # finalized on mouse-up

    win = "ROI Picker (drag=draw, n=net, s=slot, c=next cam, r=reset cam, q=save/quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, start_pt, current_rect, last_rect

        frame = frames[cam_idx]
        h, w = frame.shape[:2]

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
            current_rect = None

        elif event == cv2.EVENT_MOUSEMOVE and drawing and start_pt:
            x1, y1 = start_pt
            rx, ry, rw, rh = clamp_rect(x1, y1, x, y, w, h)
            current_rect = (rx, ry, rw, rh)

        elif event == cv2.EVENT_LBUTTONUP and drawing and start_pt:
            drawing = False
            x1, y1 = start_pt
            rx, ry, rw, rh = clamp_rect(x1, y1, x, y, w, h)
            start_pt = None
            current_rect = None
            if rw > 3 and rh > 3:
                last_rect = (rx, ry, rw, rh)

    cv2.setMouseCallback(win, on_mouse)

    def draw_overlay(frame, cam_key):
        img = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Instructions
        cv2.putText(img, f"{cam_key}", (20, 30), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "Drag to draw. n=NET, s=SLOT, c=next cam, r=reset, q=save/quit",
                    (20, 60), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Existing saved ROIs
        for label, color in [("net", (0, 255, 0)), ("slot", (255, 0, 0))]:
            r = rois[cam_key][label]
            if r:
                x, y, w, h = r
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                cv2.putText(img, label.upper(), (x, max(20, y - 10)), font, 0.9, color, 2, cv2.LINE_AA)

        # Live rubber-band rectangle while drawing
        if current_rect:
            x, y, w, h = current_rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Last finalized rectangle (candidate)
        if last_rect:
            x, y, w, h = last_rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)
            cv2.putText(img, "LAST (press n or s)", (x, min(frame.shape[0] - 10, y + h + 25)),
                        font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        return img

    while True:
        cam_key = f"camera_{cam_idx + 1}"
        cv2.imshow(win, draw_overlay(frames[cam_idx], cam_key))
        k = cv2.waitKey(20) & 0xFF

        if k == ord("n") and last_rect:
            rois[cam_key]["net"] = list(last_rect)
            print(cam_key, "net", last_rect)

        elif k == ord("s") and last_rect:
            rois[cam_key]["slot"] = list(last_rect)
            print(cam_key, "slot", last_rect)

        elif k == ord("r"):
            rois[cam_key] = {"net": None, "slot": None}
            last_rect = None
            print("reset", cam_key)

        elif k == ord("c"):
            cam_idx = (cam_idx + 1) % 2
            last_rect = None
            drawing = False
            print("switch to", f"camera_{cam_idx + 1}")

        elif k == ord("q"):
            ok = all(rois[c][k] for c in rois for k in ("net", "slot"))
            if not ok:
                print("Need NET+SLOT for both cameras before saving.")
                continue
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rois, f, indent=2)
            print("saved", out_path)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()