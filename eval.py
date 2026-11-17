import json
import re
from collections import defaultdict


def clean_string(s):

    s = str(s).strip()
    return re.sub(r'[\*\n\"\'\[\]]', '', s).lower()


def parse_llm_mcq_answer(prediction, correct_label, question_text):


    clean_pred = clean_string(prediction)
    clean_label = clean_string(correct_label)


    if clean_pred == clean_label:
        return True


    if clean_label in clean_pred:
        pattern = re.compile(r"\[(.*?)\]")
        matches = pattern.findall(question_text)
        if matches:
            options_str = matches[-1]
            options = [clean_string(opt) for opt in options_str.split("', '")]


            other_options_in_pred = [opt for opt in options if opt != clean_label and opt in clean_pred]


            if not other_options_in_pred:
                return True

    return False


def CausalDrive_Criterion_QA(dataset_json_list, model_predictions_list):


    total_images = len(dataset_json_list)
    if total_images == 0:
        return None


    sep_total_score = 0.0
    dho_total_score = 0.0
    rce_total_score = 0.0
    adr_total_score = 0.0


    tac_correct_count = 0
    tac_and_edp_action_correct_count = 0

    for d_ind, sample in enumerate(dataset_json_list):
        qa_pairs = sample['qa_pairs']
        predictions = model_predictions_list[d_ind]


        img_results = defaultdict(dict)

        for q_ind, qa in enumerate(qa_pairs):
            task_type = qa['task_type']
            category = qa['category']
            correct_label = qa['label']
            question_text = qa['question']


            pred = predictions[q_ind] if q_ind < len(predictions) else ""


            is_correct = parse_llm_mcq_answer(pred, correct_label, question_text)
            img_results[task_type][category] = is_correct


        sep_cats = img_results.get("SEP", {})
        if sep_cats:

            img_sep_score = sum(sep_cats.values()) / len(sep_cats)
            sep_total_score += img_sep_score


        dho_cats = img_results.get("DHO", {})
        if dho_cats:

            img_dho_score = 1.0 if all(dho_cats.values()) else 0.0
            dho_total_score += img_dho_score


        rce_cats = img_results.get("RCE", {})
        if rce_cats:

            row_corr = rce_cats.get("right_of_way", False)
            sig_corr = rce_cats.get("signal_constraint", False)
            img_rce_score = (0.7 * float(row_corr)) + (0.3 * float(sig_corr))
            rce_total_score += img_rce_score


        adr_cats = img_results.get("ADR", {})
        if adr_cats:

            act_corr = adr_cats.get("longitudinal_decision", False)
            foc_corr = adr_cats.get("defensive_focus", False)
            rat_corr = adr_cats.get("decision_rationale", False)
            img_adr_score = 1.0 if (act_corr and foc_corr and rat_corr) else 0.0
            adr_total_score += img_adr_score


        if "right_of_way" in rce_cats and "longitudinal_decision" in adr_cats:
            if rce_cats["right_of_way"]:
                tac_correct_count += 1
                if adr_cats["longitudinal_decision"]:
                    tac_and_edp_action_correct_count += 1


    final_sep = sep_total_score / total_images
    final_dho = dho_total_score / total_images
    final_rce = rce_total_score / total_images
    final_adr = adr_total_score / total_images

    ccc_score = tac_and_edp_action_correct_count / tac_correct_count if tac_correct_count > 0 else 0


    overall_score = (final_sep + final_dho + final_rce + final_adr) / 4.0


    report = "\n" + "=" * 65 + "\n"
    report += "🚗 CausalDrive Benchmark Evaluation Report\n"
    report += "=" * 65 + "\n\n"

    report += "[1] SEP Score (Independent Avg Accuracy)\n"
    report += f"    - Static Environment Perception:   {final_sep:.2%} \n"
    report += "      (Measures independent visual extraction capability)\n\n"

    report += "[2] DHO Score (Joint Accuracy)\n"
    report += f"    - Dynamic Hazard Observation:      {final_dho:.2%} \n"
    report += "      (Requires zero false-negatives in hazard detection)\n\n"

    report += "[3] RCE Score (Weighted Accuracy)\n"
    report += f"    - Regulatory Constraint Eval:      {final_rce:.2%} \n"
    report += "      (Weights: 70% Right-of-Way, 30% Signal Constraint)\n\n"

    report += "[4] ADR Score (Strict Synergy Score)\n"
    report += f"    - Action & Decision Rationale:     {final_adr:.2%} \n"
    report += "      (Action + Focus + Rationale must ALL be correct)\n\n"

    report += "-" * 65 + "\n"
    report += "[★] CCC Score (Causal Chain Consistency)\n"
    report += f"    - P(Action Correct | Right-of-Way Correct): {ccc_score:.2%} ({tac_and_edp_action_correct_count}/{tac_correct_count})\n"
    report += "      (Indicates decisions grounded in actual legal reasoning)\n\n"

    report += "[🏆] Overall CausalDrive Score (Unweighted Avg)\n"
    report += f"    - Final Comprehensive Score:       {overall_score * 100:.2f} / 100\n"
    report += "=" * 65 + "\n"

    print(report)
    return {
        "SEP": final_sep,
        "DHO": final_dho,
        "RCE": final_rce,
        "ADR": final_adr,
        "CCC": ccc_score,
        "Overall": overall_score
    }


if __name__ == "__main__":
    pass