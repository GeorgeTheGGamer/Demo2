def get_objects(frame, model, conf_thres=0.3):
    """
    Read objects from a frame, return boxes details
    :param model: used YOLO model
    :param frame: input frame
    :param conf_thres: threshold of confidence rate for detecting objects
    :return: all details in a dictionary {cls,conf,bbox}
    """
    results = model(frame, verbose=False)[0]
    res_list = []
    for box in results.boxes:
        # get box details
        cls = int(box.cls[0].item())
        conf = float(box.conf)
        if conf < conf_thres:
            continue

        x1,y1,x2,y2 = box.xyxy[0].tolist()
        res_list.append({
            "cls": cls,
            "conf": conf,
            "bbox": [x1, y1, x2, y2],
        })
    return res_list