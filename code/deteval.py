import math
from collections import namedtuple
from copy import deepcopy

import numpy as np


def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
        'AREA_RECALL_CONSTRAINT' : 0.8,
        'AREA_PRECISION_CONSTRAINT' : 0.4,
        'EV_PARAM_IND_CENTER_DIFF_THR': 1,
        'MTYPE_OO_O':1.,
        'MTYPE_OM_O':0.8,
        'MTYPE_OM_M':1.,
        'GT_SAMPLE_NAME_2_ID':'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID':'res_img_([0-9]+).txt',
        'CRLF':False # Lines are delimited by Windows CRLF format
    }


def calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict,
                         eval_hparams=None, bbox_format='rect', verbose=False):
    """
    현재는 rect(xmin, ymin, xmax, ymax) 형식의 bounding box만 지원함. 다른 형식(quadrilateral,
    poligon, etc.)의 데이터가 들어오면 외접하는 rect로 변환해서 이용하고 있음.
    """

    def one_to_one_match(row, col):
        cont = 0
        for j in range(len(recallMat[0])):
            if recallMat[row,j] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and precisionMat[row,j] >= eval_hparams['AREA_PRECISION_CONSTRAINT'] :
                cont = cont +1
        if (cont != 1):
            return False
        cont = 0
        for i in range(len(recallMat)):
            if recallMat[i,col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and precisionMat[i,col] >= eval_hparams['AREA_PRECISION_CONSTRAINT'] :
                cont = cont +1
        if (cont != 1):
            return False

        if recallMat[row,col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and precisionMat[row,col] >= eval_hparams['AREA_PRECISION_CONSTRAINT'] :
            return True
        return False

    def one_to_one_match_v2(row, col):
        if row_sum[row] != 1:
            return False

        if col_sum[col] != 1:
            return False

        if recallMat[row, col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and \
                precisionMat[row, col] >= eval_hparams['AREA_PRECISION_CONSTRAINT']:
            return True
        return False

    def num_overlaps_gt(gtNum):
        cont = 0
        for detNum in range(len(detRects)):
            if recallMat[gtNum,detNum] > 0 :
                cont = cont +1
        return cont

    def num_overlaps_det(detNum):
        cont = 0
        for gtNum in range(len(recallMat)):
            if recallMat[gtNum,detNum] > 0 :
                cont = cont +1
        return cont

    def is_single_overlap(row, col):
        if num_overlaps_gt(row)==1 and num_overlaps_det(col)==1:
            return True
        else:
            return False

    def one_to_many_match(gtNum):
        many_sum = 0
        detRects = []
        for detNum in range(len(recallMat[0])):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0:
                if precisionMat[gtNum,detNum] >= eval_hparams['AREA_PRECISION_CONSTRAINT'] :
                    many_sum += recallMat[gtNum,detNum]
                    detRects.append(detNum)
        if round(many_sum,4) >=eval_hparams['AREA_RECALL_CONSTRAINT'] :
            return True,detRects
        else:
            return False,[]

    def many_to_one_match(detNum):
        many_sum = 0
        gtRects = []
        for gtNum in range(len(recallMat)):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0:
                if recallMat[gtNum,detNum] >= eval_hparams['AREA_RECALL_CONSTRAINT'] :
                    many_sum += precisionMat[gtNum,detNum]
                    gtRects.append(gtNum)
        if round(many_sum,4) >=eval_hparams['AREA_PRECISION_CONSTRAINT'] :
            return True,gtRects
        else:
            return False,[]

    def area(a, b):
            dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin) + 1
            dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin) + 1
            if (dx>=0) and (dy>=0):
                    return dx*dy
            else:
                    return 0.

    def center(r):
        x = float(r.xmin) + float(r.xmax - r.xmin + 1) / 2.
        y = float(r.ymin) + float(r.ymax - r.ymin + 1) / 2.
        return Point(x,y)

    def point_distance(r1, r2):
        distx = math.fabs(r1.x - r2.x)
        disty = math.fabs(r1.y - r2.y)
        return math.sqrt(distx * distx + disty * disty )


    def center_distance(r1, r2):
        return point_distance(center(r1), center(r2))

    def diag(r):
        w = (r.xmax - r.xmin + 1)
        h = (r.ymax - r.ymin + 1)
        return math.sqrt(h * h + w * w)

    if eval_hparams is None:
        eval_hparams = default_evaluation_params()

    if bbox_format != 'rect':
        raise NotImplementedError

    # bbox들이 rect 이외의 형식으로 되어있는 경우 rect 형식으로 변환
    _pred_bboxes_dict, _gt_bboxes_dict= deepcopy(pred_bboxes_dict), deepcopy(gt_bboxes_dict)
    pred_bboxes_dict, gt_bboxes_dict = dict(), dict()
    for sample_name, bboxes in _pred_bboxes_dict.items():
        # 원래 rect 형식이었으면 변환 없이 그대로 이용
        if len(bboxes) > 0 and np.array(bboxes[0]).ndim == 1 and len(bboxes[0]) == 4:
            pred_bboxes_dict = _pred_bboxes_dict
            break

        pred_bboxes_dict[sample_name] = []
        for bbox in map(np.array, bboxes):
            rect = [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max(), bbox[:, 1].max()]
            pred_bboxes_dict[sample_name].append(rect)
    for sample_name, bboxes in _gt_bboxes_dict.items():
        # 원래 rect 형식이었으면 변환 없이 그대로 이용
        if len(bboxes) > 0 and np.array(bboxes[0]).ndim == 1 and len(bboxes[0]) == 4:
            gt_bboxes_dict = _gt_bboxes_dict
            break

        gt_bboxes_dict[sample_name] = []
        for bbox in map(np.array, bboxes):
            rect = [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max(), bbox[:, 1].max()]
            gt_bboxes_dict[sample_name].append(rect)

    perSampleMetrics = {}

    methodRecallSum = 0
    methodPrecisionSum = 0

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    Point = namedtuple('Point', 'x y')

    numGt = 0
    numDet = 0

    for sample_name in gt_bboxes_dict:

        recall = 0
        precision = 0
        hmean = 0
        recallAccum = 0.
        precisionAccum = 0.
        gtRects = []
        detRects = []
        gtPolPoints = []
        detPolPoints = []
        pairs = []
        evaluationLog = ""

        recallMat = np.empty([1, 1])
        precisionMat = np.empty([1, 1])

        pointsList = gt_bboxes_dict[sample_name]

        for n in range(len(pointsList)):
            points = pointsList[n]
            gtRect = Rectangle(*points)
            gtRects.append(gtRect)
            gtPolPoints.append(np.array(points).tolist())

        evaluationLog += "GT rectangles: " + str(len(gtRects)) + '\n'

        if sample_name in pred_bboxes_dict:
            pointsList = pred_bboxes_dict[sample_name]

            for n in range(len(pointsList)):
                points = pointsList[n]
                detRect = Rectangle(*points)
                detRects.append(detRect)
                detPolPoints.append(np.array(points).tolist())

            evaluationLog += "DET rectangles: " + str(len(detRects)) + '\n'

            if len(gtRects)==0: # gt가 없으면 진입
                recall = 1
                precision = 0 if len(detRects)>0 else 1

            if len(detRects)>0: # 예측 박스가 있으면 if문 진입
                #Calculate recall and precision matrixs
                outputShape=[len(gtRects),len(detRects)] # gt수, det수
                recallMat = np.empty(outputShape) # gt와 det의 area recall 기록
                precisionMat = np.empty(outputShape) # gt와 det의 area precision 기록
                gtRectMat = np.zeros(len(gtRects),np.int8) # gt 개수 세기 위함
                detRectMat = np.zeros(len(detRects),np.int8) # det 개수 세기 위함
                for gtNum in range(len(gtRects)): 
                    for detNum in range(len(detRects)):
                        rG = gtRects[gtNum] 
                        rD = detRects[detNum]
                        intersected_area = area(rG,rD) # intersect area 계산
                        rgDimensions = ( (rG.xmax - rG.xmin+1) * (rG.ymax - rG.ymin+1) )
                        rdDimensions = ( (rD.xmax - rD.xmin+1) * (rD.ymax - rD.ymin+1))
                        recallMat[gtNum,detNum] = 0 if rgDimensions==0 else  intersected_area / rgDimensions # gt 박스 넓이가 0이면 area recall도 0
                        precisionMat[gtNum,detNum] = 0 if rdDimensions==0 else intersected_area / rdDimensions # det 박스 넓이가 0이면 area precision도 0

                recall_cond = recallMat >= eval_hparams['AREA_RECALL_CONSTRAINT'] # area recall 제한 값인 0.8 이상 
                precision_cond = precisionMat >= eval_hparams['AREA_PRECISION_CONSTRAINT'] # area precision 제한 값인 0.4 이상
                cond = recall_cond & precision_cond # 비트 연산을 통해서 제한 값 둘다 통과하는 것 파악
                col_sum = np.sum(cond, axis=0)  # col_sum은 det 박스 별 기준치를 통과해서 매칭되는 gt가 몇개냐
                row_sum = np.sum(cond, axis=1) # row_sum은 gt 박스 별 기준치를 통과해서 매칭되는 det가 몇개냐 

                # Find one-to-one matches
                # 하나의 gt에 하나의 det가 매칭된 것 검사
                evaluationLog += "Find one-to-one matches\n"
                for gtNum in range(len(gtRects)):
                    for detNum in range(len(detRects)):
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0: # 아직 매칭되지 않은 것들만 진행
                            # match = one_to_one_match_v2(gtNum, detNum) 설명
                            # row_sum[gtNum], col_sum[detNum]이 1이 아니면 False -> 기준치를 통과하는 매칭이 없기 때문
                            # 둘다 1이면 기준치를 통과한 매칭이 하나씩 있는 것
                            # 그리고 매칭되는 부분이 recallMat[gtNum, detNum]과 precisionMat[gtNum, detNum]을 이용해서 
                            # [gtNum, detNum]이 위에서 검출한 기준치 통과 매칭에 해당하는지 확인
                            match = one_to_one_match_v2(gtNum, detNum) 
                            if match is True :
                                #in deteval we have to make other validation before mark as one-to-one
                                # is_single_overlap(gtNum, detNum) 설명
                                # recallMat을 통해서 gtNum과 area recall 값이 0 이상인 det의 수를 카운팅
                                # recallMat을 통해서 detNum과 area recall 값이 0 이상인 gt의 수를 카운팅
                                # 만약 둘 중 하나라도 1이 아니라면 False
                                # 왜냐하면 해당 값들이 기준치 점수에 미달이지만 one to one 매칭이 아니기 때문
                                if is_single_overlap(gtNum, detNum) is True :
                                    rG = gtRects[gtNum]
                                    rD = detRects[detNum]
                                    normDist = center_distance(rG, rD) # rG, rD의 중심점 계산 후, 두 중심점간의 거리 계산
                                    normDist /= diag(rG) + diag(rD) # rG와 rD의 각 대각선을 구해서 더한 값으로 중심점 간의 거리를 나눈다.
                                    normDist *= 2.0 # 정규화된 거리를 2배 확장
                                    if normDist < eval_hparams['EV_PARAM_IND_CENTER_DIFF_THR'] : # 1이하면 진입
                                        gtRectMat[gtNum] = 1 # 해당 gt는 매칭되었기 때문에 1로 변환해서 다음 반복문에서 필터링할 수 있도록 하낟.
                                        detRectMat[detNum] = 1 # 해당 det는 매칭되었기 때문에 1로 변환해서 다음 반복문에서 필터링할 수 있도록 하낟.
                                        recallAccum += eval_hparams['MTYPE_OO_O'] # 1
                                        precisionAccum += eval_hparams['MTYPE_OO_O'] # 1
                                        pairs.append({'gt':gtNum,'det':detNum,'type':'OO'}) # 매칭된 gt와 det를 matching type과 함께 기록
                                        evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"
                                    else:
                                        evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(detNum) + " normDist: " + str(normDist) + " \n"
                                else:
                                    evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(detNum) + " not single overlap\n"

                # Find one-to-many matches
                # 하나의 gt에 여러개의 det가 매칭된 것을 검사
                evaluationLog += "Find one-to-many matches\n"
                for gtNum in range(len(gtRects)):
                    # one_to_many_match(gtNum) 설명
                    # det의 개수만큼 반복문을 수행하면서
                    # gtRectMat[gtNum]과 detRectMat[detNum]이 아직 매칭되지 않은 것을 찾는다. 
                    # gtNum이 이미 매칭이 되었다면 바로 False
                    # 아직 매칭되지 않은 det가 있다면 precisionMat[gtNum, detNum]의 값이 0.4 이상인지 확인
                    # 0.4 이상이면 recallMat[gtNum, detNum]의 값을 더해주고, detNum을 gtNum과 매칭된 det들을 모은 리스트에 추가
                    # 반복문을 다 돌고, recall을 더한 값을 소수점 4자리까지 반올림한 값이 0.8이상이면 True와 함께 매칭된 det들을 반환
                    match,matchesDet = one_to_many_match(gtNum)
                    if match is True :
                        evaluationLog += "num_overlaps_gt=" + str(num_overlaps_gt(gtNum))
                        #in deteval we have to make other validation before mark as one-to-one
                        # num_overlaps_gt(gtNum)>=2 설명
                        # gtNum과 area recall값이 0이 넘는 det의 수를 반환하는 함수
                        # 일대다 매칭이기 때문에 2개 이상 매칭되어야 if문 진입
                        if num_overlaps_gt(gtNum)>=2 :
                            gtRectMat[gtNum] = 1 # 해당 gt는 일대다 매칭되었기 때문에 1로 표시
                            # matchesDet의 길이가 1이면 gtNum과 area recall값이 0이 넘는 것들이 2개 이상이 있지만, 
                            # (아직 매칭되지 않은 det가 1개 or Area Precision 제한인 0.4 이상을 넘는 det가 1개) and (하나 밖에 없는 det가 Area Recall 제한인 0.8 이상을 충족)
                            # 위의 조건을 통과한 상황이다. 그렇기 때문에 one to one match로 보고 1점
                            # 그렇지 않은 경우는 0.8점 (일대다 매칭은 패널티로 0.8)
                            recallAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesDet)==1 else eval_hparams['MTYPE_OM_O'])

                            # precision은 예측에 성공한 predict box가 여러개 이므로 matchesDet 길이만큼 늘어나야 한다.
                            precisionAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesDet)==1 else eval_hparams['MTYPE_OM_O']*len(matchesDet))

                            # 매칭된 gt와 det를 기록, 위의 설명과 동일하게 matchesDet가 1이면 one to one match 아니면 one to many match
                            pairs.append({'gt':gtNum,'det':matchesDet,'type': 'OO' if len(matchesDet)==1 else 'OM'})

                            # 매칭된 det들을 모두 1로 표시
                            for detNum in matchesDet :
                                detRectMat[detNum] = 1
                            evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(matchesDet) + "\n"
                        else:
                            evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(matchesDet) + " not single overlap\n"

                # Find many-to-one matches
                # 여러개의 gt에 하나의 det가 매칭된 것 검사
                evaluationLog += "Find many-to-one matches\n"
                for detNum in range(len(detRects)):
                    # many_to_one_match(detNum) 설명
                    # gt의 개수만큼 반복문을 수행하면서
                    # gtRectMat[gtNum]과 detRectMat[detNum]이 아직 매칭되지 않은 것을 찾는다. 
                    # detNum이 이미 매칭이 되었다면 바로 False
                    # 아직 매칭되지 않은 gt가 있다면 recallMat[gtNum, detNum]의 값이 0.8 이상인지 확인
                    # 0.8 이상이면 precisionMat[gtNum, detNum]의 값을 더해주고, gtNum을 detNum과 매칭된 gt들을 모은 리스트에 추가
                    # 반복문을 다 돌고, precision을 더한 값을 소수점 4자리까지 반올림한 값이 0.4이상이면 True와 함께 매칭된 gt들을 반환
                    match,matchesGt = many_to_one_match(detNum)
                    if match is True :
                        #in deteval we have to make other validation before mark as one-to-one
                        # num_overlaps_det(detNum)>=2 설명
                        # detNum과 area recall값이 0이 넘는 gt의 수를 반환하는 함수
                        # 다대일 매칭이기 때문에 2개 이상 매칭되어야 if문 진입
                        if num_overlaps_det(detNum)>=2 :
                            detRectMat[detNum] = 1 # 해당 det는 다대일 매칭되었기 때문에 1로 표시

                            # matchesGt의 길이가 1이면 detNum과 area recall값이 0이 넘는 것들이 2개 이상이 있지만, 
                            # (아직 매칭되지 않은 gt가 1개 or Area Recall 제한인 0.8 이상을 넘는 gt가 1개) and (하나 밖에 없는 gt가 Area Precision 제한인 0.4 이상을 충족)
                            # 위의 조건을 통과한 상황이다. 그렇기 때문에 one to one match로 보고 1점
                            # 그렇지 않은 경우에도 1점이다. 하지만 prediction과 매칭된 gt가 여러개 이므로 recall은 matchesGt 길이만큼 늘어나야 한다.
                            recallAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesGt)==1 else eval_hparams['MTYPE_OM_M']*len(matchesGt))
                            # matchesGt의 길이가 1인 것과 상관없이 똑같이 1점이 더해진다. 예측된 박스는 1개이기 때문
                            precisionAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesGt)==1 else eval_hparams['MTYPE_OM_M'])

                            # 매칭된 gt와 det를 기록, 위의 설명과 동일하게 matchesGt가 1이면 one to one match 아니면 many to one match
                            pairs.append({'gt':matchesGt,'det':detNum,'type': 'OO' if len(matchesGt)==1 else 'MO'})

                            # 매칭된 gt 모두 1로 표시
                            for gtNum in matchesGt :
                                gtRectMat[gtNum] = 1
                            evaluationLog += "Match GT #" + str(matchesGt) + " with Det #" + str(detNum) + "\n"
                        else:
                            evaluationLog += "Match Discarded GT #" + str(matchesGt) + " with Det #" + str(detNum) + " not single overlap\n"

                numGtCare = len(gtRects)
                # gt가 없으면
                if numGtCare == 0: # 이미 위에서 했지만 다시
                    recall = float(1)
                    precision = float(0) if len(detRects)>0 else float(1)
                else: # gt가 있으면 
                    recall = float(recallAccum) / numGtCare # 축적한 recall / gt수 
                    # 예측 박스 없으면 precision 0, 있으면 축적한 precision / det수 
                    precision =  float(0) if len(detRects)==0 else float(precisionAccum) / len(detRects) 
                hmean = 0 if (precision + recall)==0 else 2.0 * precision * recall / (precision + recall)

        methodRecallSum += recallAccum # 해당 이미지의 recallAccum 점수
        methodPrecisionSum += precisionAccum # 해당 이미지의 precisionAccum 점수
        numGt += len(gtRects)
        numDet += len(detRects)

        perSampleMetrics[sample_name] = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'recall_matrix': [] if len(detRects)>100 else recallMat.tolist(),
            'precision_matrix': [] if len(detRects)>100 else precisionMat.tolist(),
            'gt_bboxes': gtPolPoints,
            'det_bboxes': detPolPoints,
        }

        if verbose:
            perSampleMetrics[sample_name].update(evaluation_log=evaluationLog)

    methodRecall = 0 if numGt==0 else methodRecallSum/numGt
    methodPrecision = 0 if numDet==0 else methodPrecisionSum/numDet
    methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)

    methodMetrics = {'precision': methodPrecision, 'recall': methodRecall,'hmean': methodHmean}

    resDict = {'calculated': True, 'Message': '', 'total': methodMetrics,
               'per_sample': perSampleMetrics, 'eval_hparams': eval_hparams}

    return resDict
