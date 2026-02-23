def build_fcpxml(markers, fps: int, project_name: str):
    if not markers:
        raise ValueError("No markers to write.")

    last_frame = markers[-1]["frame"]
    tail = fps * 5  # 5s padding
    total_frames = last_frame + tail

    frame_duration = f"1/{fps}s"

    xml = []
    xml.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml.append('<fcpxml version="1.4">')
    xml.append('  <resources>')
    xml.append(
        f'    <format id="r1" name="FFVideoFormat1080p{fps}" frameDuration="{frame_duration}" '
        f'width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>'
    )
    xml.append('  </resources>')
    xml.append('  <library>')
    xml.append('    <event name="markers">')
    xml.append(f'      <project name="{escape(project_name)}">')
    xml.append(f'        <sequence format="r1" duration="{t_rational(total_frames, fps)}" tcStart="0s" tcFormat="NDF">')
    xml.append('          <spine>')
    xml.append(f'            <gap name="Markers" offset="0s" start="0s" duration="{t_rational(total_frames, fps)}">')

    for m in markers:
        note = (m.get("note") or "").strip()
        color_tag = (m.get("color_in") or "Blue").strip()
        prefix = f"[{color_tag}]"
        note2 = f"{prefix} {note}".strip() if note else prefix

        xml.append(
            f'              <marker start="{t_rational(m["frame"], fps)}" '
            f'duration="{t_rational(m["duration_frames"], fps)}" '
            f'value="{escape(m["name"])}" note="{escape(note2)}"/>'
        )

    xml.append('            </gap>')
    xml.append('          </spine>')
    xml.append('        </sequence>')
    xml.append('      </project>')
    xml.append('    </event>')
    xml.append('  </library>')
    xml.append('</fcpxml>')
    return "\n".join(xml) + "\n"