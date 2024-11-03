# make a paragraph out of the word images
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from typing import Any

Document = dict[str : Image.Image, str : list[dict]]


def make_document(words, width=2800, height=2800, max_retries=3) -> Document:
    for attempt in range(max_retries):
        try:
            image = Image.fromarray(
                np.random.normal(230, 6, (height, width, 3)).astype(np.uint8)
            )
            
            # 배치된 bbox들을 저장할 리스트
            placed_bboxes = []
            
            for word in words:
                patch = word["patch"]
                size = word["size"]
                m = word["margins"]
                
                max_attempts = 100
                placed = False
                
                # 이미지가 너무 큰 경우 다시 생성 요청
                if size[0] > width or size[1] > height:
                    raise ValueError("이미지가 너무 큽니다.")
                
                # 랜덤한 위치에 배치 시도
                for _ in range(max_attempts):
                    x = np.random.randint(0, max(1, width - size[0]))
                    y = np.random.randint(0, max(1, height - size[1]))
                    
                    # 현재 bbox 계산
                    vs = (x + m[0], y + m[1], x + size[0] - m[2], y + size[1] - m[3])
                    current_bbox = [vs[0], vs[1], vs[2], vs[1], vs[2], vs[3], vs[0], vs[3]]
                    
                    # bbox 충돌 검사
                    overlap = False
                    for placed_bbox in placed_bboxes:
                        if check_overlap(current_bbox, placed_bbox):
                            overlap = True
                            break
                    
                    if not overlap:
                        word["bbox"] = current_bbox
                        image.paste(patch, (x, y))
                        placed_bboxes.append(current_bbox)
                        placed = True
                        break
                
                if not placed:
                    raise Exception("이미지를 배치할 공간이 부족합니다.")
                    
            return {"image": image, "words": words}
            
        except (ValueError, Exception) as e:
            if attempt == max_retries - 1:  # 마지막 시도였다면
                raise Exception(f"문서 생성 실패: {str(e)}")
            continue  # 다시 시도


def check_overlap(bbox1, bbox2):
    """두 bbox가 겹치는지 확인"""
    x1_min, y1_min = min(bbox1[0], bbox1[4]), min(bbox1[1], bbox1[5])
    x1_max, y1_max = max(bbox1[0], bbox1[4]), max(bbox1[1], bbox1[5])
    
    x2_min, y2_min = min(bbox2[0], bbox2[4]), min(bbox2[1], bbox2[5])
    x2_max, y2_max = max(bbox2[0], bbox2[4]), max(bbox2[1], bbox2[5])
    
    return not (x1_max < x2_min or x2_max < x1_min or 
               y1_max < y2_min or y2_max < y1_min)


def draw_bbox(document: Document, ax=None) -> None:
    ax = ax or plt
    for word in document["words"]:
        bbox = word["bbox"]
        xs = [bbox[i] for i in range(0, 8, 2)] + [bbox[0]]
        ys = [bbox[i] for i in range(1, 8, 2)] + [bbox[1]]
        ax.plot(xs, ys, color="blue")
    ax.imshow(document["image"])
    ax.axis("off")


def pad_document_inplace(document: Document, pad=50, color=None) -> Document:
    """pad the document image and update the bounding boxes in-place."""
    if color is None:
        color = [64, 64, 64]
    image = cv2.copyMakeBorder(
        np.array(document["image"]),
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=[64, 64, 64],
    )
    document["image"] = Image.fromarray(image)
    for word in document["words"]:
        word["bbox"] = [v + pad for v in word["bbox"]]
    return document


def partial_copy(document: Document) -> Document:
    """
    copy some of the compoents only,
    i.e. the whole document image and bounding boxes.
    the rest are shared references.
    """
    image = document["image"].copy()
    words = [word.copy() for word in document["words"]]
    for word in words:
        word["bbox"] = word["bbox"].copy()
    return {"image": image, "words": words}


def simple_shows(args: list[dict[str:Any]], plots_per_row=4) -> None:
    len_args = len(args)
    fig_shape = (len_args // plots_per_row, plots_per_row)
    _ , axs = plt.subplots(*fig_shape, figsize=(3 * fig_shape[1], 4 * fig_shape[0]))
    for i in range(len_args):
        row, col = i // plots_per_row, i % plots_per_row
        draw_bbox(args[i]["doc"], axs[row][col])
        axs[row][col].set_title(args[i]["title"])
    plt.tight_layout()
    plt.show()