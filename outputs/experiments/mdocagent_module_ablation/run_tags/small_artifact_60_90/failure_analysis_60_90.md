# Failure Analysis: small_artifact_60_90

## Summary

- original_only: 0.466667
- original_plus_artifact: 0.400000
- artifact_only: 0.266667
- delta original_plus_artifact vs original_only: -0.066667

## Win/Loss Buckets

| bucket | count | page changed | page unchanged |
| --- | ---: | ---: | ---: |
| wrong_to_right | 1 | 1 | 0 |
| right_to_wrong | 3 | 3 | 0 |
| both_right | 11 | 9 | 2 |
| both_wrong | 15 | 12 | 3 |

## Loss/Win Signals

Loss reason counters for `original_only right -> original_plus_artifact wrong`:
- page_selection_changed: 3
- page_order_only_changed: 1
- text_branch_changed: 3
- text_image_branches_collapsed: 3
- no_nonzero_artifact_score_for_doc_candidates: 3
- plus_pages_no_artifacts: 3
- page_set_changed: 2
- gold_page_dropped: 1

Win reason counters for `original_only wrong -> original_plus_artifact right`:
- page_selection_changed: 1
- page_set_changed: 1
- no_nonzero_artifact_score_for_doc_candidates: 1

Artifact score coverage by bucket:
- both_wrong has_nonzero_artifact_score=False: 15
- both_right has_nonzero_artifact_score=False: 11
- right_to_wrong has_nonzero_artifact_score=False: 3
- wrong_to_right has_nonzero_artifact_score=False: 1

## Gold Page Diagnostics

| bucket | orig gold any | plus gold any | gold dropped | gold added |
| --- | ---: | ---: | ---: | ---: |
| wrong_to_right | 1 | 0 | 1 | 0 |
| right_to_wrong | 1 | 0 | 1 | 0 |
| both_right | 7 | 6 | 1 | 0 |
| both_wrong | 8 | 7 | 1 | 0 |

## Cases

### idx 64 - right_to_wrong
- doc: `2310.05634v2.pdf`
- question: How many tables include "F1" as a metric?
- gold_pages: [7, 9]
- original_pages: [6, 8, 4, 2] correct=True
- plus_pages: [6, 4, 8, 2] correct=False
- page_set_changed=False page_order_only_changed=True has_nonzero_artifact_score=False
- original_by_key: {'image-top-10-question': [6, 8, 4, 2], 'text-top-10-question': [6, 6, 6, 4]}
- plus_by_key: {'image-top-10-question': [6, 4, 8, 2], 'text-top-10-question': [6, 4, 8, 2]}
- added_pages: [] dropped_pages: []
- original_gold_any=False plus_gold_any=False
- added_artifacts: {'num_artifacts': 0, 'pages_with_artifacts': [], 'artifact_types': {}, 'modalities': {}, 'anchor_types': {}}
- original_answer: 3
- plus_answer: 2

### idx 70 - right_to_wrong
- doc: `2401.18059v1.pdf`
- question: What are the horizontal and vertical axis of Figure 3 respectively?
- gold_pages: [6]
- original_pages: [15, 5, 14, 21, 1, 4, 8] correct=True
- plus_pages: [15, 5, 14, 21] correct=False
- page_set_changed=True page_order_only_changed=False has_nonzero_artifact_score=False
- original_by_key: {'image-top-10-question': [15, 5, 14, 21], 'text-top-10-question': [1, 21, 4, 8]}
- plus_by_key: {'image-top-10-question': [15, 5, 14, 21], 'text-top-10-question': [15, 5, 14, 21]}
- added_pages: [] dropped_pages: [1, 4, 8]
- original_gold_any=False plus_gold_any=False
- added_artifacts: {'num_artifacts': 0, 'pages_with_artifacts': [], 'artifact_types': {}, 'modalities': {}, 'anchor_types': {}}
- original_answer: The horizontal axis (x-axis) represents Context Length, ranging from 0 to 2500, while the vertical axis (y-axis) represents Performance, specifically the F1 score, ranging from 0 to 100. This is consistent across all agents, with the Text Agent and General Agent confirming the performance metrics include accuracy or BLEU/ROUGE/METEOR scores, and the Image Agent directly identifying the y-axis label as 'F1'.
- plus_answer: The horizontal axis is 'Context Length' and the vertical axis is 'Accuracy'.

### idx 71 - right_to_wrong
- doc: `2401.18059v1.pdf`
- question: Based on "PSEUDOCODE FOR RETRIEVAL METHODS", which algorithm has more number of lines? (Give the algorithm name)
- gold_pages: [1, 8, 1, 9]
- original_pages: [17, 0, 18, 2, 1, 3] correct=True
- plus_pages: [17, 0, 18, 2] correct=False
- page_set_changed=True page_order_only_changed=False has_nonzero_artifact_score=False
- original_by_key: {'image-top-10-question': [17, 0, 18, 2], 'text-top-10-question': [17, 1, 3, 1]}
- plus_by_key: {'image-top-10-question': [17, 0, 18, 2], 'text-top-10-question': [17, 0, 18, 2]}
- added_pages: [] dropped_pages: [1, 3]
- original_gold_any=True plus_gold_any=False
- added_artifacts: {'num_artifacts': 0, 'pages_with_artifacts': [], 'artifact_types': {}, 'modalities': {}, 'anchor_types': {}}
- original_answer: Collapsed Tree Algorithm
- plus_answer: Algorithm 2

### idx 80 - wrong_to_right
- doc: `2312.10997v5.pdf`
- question: According to table II, which are the datasets that has exactly three methods?
- gold_pages: [1, 3]
- original_pages: [5, 12, 13, 14, 1, 11] correct=False
- plus_pages: [12, 5, 13, 14] correct=True
- page_set_changed=True page_order_only_changed=False has_nonzero_artifact_score=False
- original_by_key: {'image-top-10-question': [5, 12, 13, 14], 'text-top-10-question': [12, 1, 5, 11]}
- plus_by_key: {'image-top-10-question': [12, 5, 13, 14], 'text-top-10-question': [12, 5, 13, 14]}
- added_pages: [] dropped_pages: [1, 11]
- original_gold_any=True plus_gold_any=False
- added_artifacts: {'num_artifacts': 0, 'pages_with_artifacts': [], 'artifact_types': {}, 'modalities': {}, 'anchor_types': {}}
- original_answer: None of the agents provided a consistent or correct answer. The General Agent incorrectly listed HotpotQA, 2WikiMultiHopQA, and MuSiQue as having exactly three methods, but these are multi-hop datasets and not necessarily the ones with exactly three methods as per the table. The Text Agent listed PopQA, MS MARCO, and SST-2, which may be correct based on their own interpretation, but without access to the actual Table II, this cannot be verified. The Image Agent listed TriviaQA, WebQ, and MS MARCO, which are single-hop datasets. Since the question requires identifying datasets with exactly three methods from Table II, and no agent provided a universally consistent or verifiable answer, no final decision can be made with confidence. Therefore, the correct response is that no consensus or reliable answer exists based on the provided information.
- plus_answer: PopQA, MS MARCO, and SST-2

