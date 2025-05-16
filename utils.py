def apk(actual, predicted, k=50):
    """Average Precision at k"""
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1.0
            score += hits / (i + 1.0)
    return score / min(len(actual), k) if actual else 0.0

def mapk(actual_list, predicted_list, k=50):
    """Mean Average Precision at k"""
    return sum(apk(a, p, k) for a, p in zip(actual_list, predicted_list)) / len(actual_list)