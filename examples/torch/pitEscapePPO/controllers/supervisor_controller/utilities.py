import numpy as np


def getDistanceFromCenter(node):
    position = node.getPosition()
    # Zeros are the center coordinates
    distanceX = position[0] - 0
    distanceZ = position[2] - 0
    return (distanceX * distanceX + distanceZ * distanceZ)**0.5  # sqrt


def normalizeToRange(value, minVal, maxVal, newMin, newMax, clip=False):
    """
    Normalize value to a specified new range by supplying the current range.

    :param value: value to be normalized
    :param minVal: value's min value, value ∈ [minVal, maxVal]
    :param maxVal: value's max value, value ∈ [minVal, maxVal]
    :param newMin: normalized range min value
    :param newMax: normalized range max value
    :param clip: whether to clip normalized value to new range or not
    :return: normalized value ∈ [newMin, newMax]
    """
    value = float(value)
    minVal = float(minVal)
    maxVal = float(maxVal)
    newMin = float(newMin)
    newMax = float(newMax)

    if clip:
        return np.clip((newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax, newMin, newMax)
    else:
        return (newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax
