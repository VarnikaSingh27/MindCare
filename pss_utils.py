# pss_utils.py
def compute_pss_level(pss_answers):
    """
    pss_answers: list of 10 integers (0-4)
    Returns (total_score, level_string, reversed_answers)
    reverse indices: 3,4,6,7 according to your script (0-based)
    """
    if not isinstance(pss_answers, (list,tuple)) or len(pss_answers) != 10:
        raise ValueError("pss_answers must be a list of 10 integers (0-4)")
    answers = [int(x) for x in pss_answers]
    # reverse scoring for indices 3,4,6,7 (0-based)
    reverse_indices = [3,4,6,7]
    for idx in reverse_indices:
        answers[idx] = 4 - answers[idx]
    total = sum(answers)
    if total <= 13:
        level = "Low"
    elif total <= 26:
        level = "Moderate"
    else:
        level = "High"
    return total, level, answers