# Hurt Artifact Diagnosis

## Record 187
Doc: `2303.08559v2.pdf`
Question: What is the performance of filter-then-rerank methods (w.o. ensemble) on 50-shot CONLL03 dataset?
Original answer: The performance for the 50-shot CONLL03 dataset is not provided in the visible tables.
Original+artifact answer: 77.2
- branch `image-top-10-question` original [8, 7, 5, 16] -> original_plus [8, 7, 5, 16]
  - page 8: top `text_span` weight=1.0, raw=8.652, weighted=8.652, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='Table 3: Overall results of LLM-based ICL methods, SLM-based supervised methods, and our proposed filter-then- rerank (S'
  - page 7: top `section_header` weight=1.0, raw=1.953, weighted=1.953, element_locatable=None, anchors=['text_block'], locators=['source_block', 'section_block', 'text_offset'], content='Filter'
  - page 14: top `caption` weight=1.0, raw=0.000, weighted=0.000, element_locatable=None, anchors=['text_block'], locators=['source_block', 'caption_block', 'text_offset'], content='Mock caption candidate anchored to a text block.'
- branch `text-top-10-question` original [7, 8, 8, 7] -> original_plus [8, 7, 16, 6]
  - page 7: top `section_header` weight=1.0, raw=1.537, weighted=1.537, element_locatable=None, anchors=['text_block'], locators=['source_block', 'section_block', 'text_offset'], content='Filter'
  - page 8: top `text_span` weight=1.0, raw=7.242, weighted=7.242, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='Table 3: Overall results of LLM-based ICL methods, SLM-based supervised methods, and our proposed filter-then- rerank (S'

## Record 207
Doc: `05-03-18-political-release.pdf`
Question: Among the adults conducted the survey on April 25 - May 1 2022, how many adults rated Trump's government ethical standards as poor? 
Original answer: The information is not available in the image.
Original+artifact answer: Approximately 540 adults rated Trump's government ethical standards as poor.
- branch `image-top-10-question` original [2, 7, 3, 8] -> original_plus [2, 10, 3, 7]
  - page 2: top `text_span` weight=1.0, raw=6.356, weighted=6.356, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='PEW RESEARCH CENTER www.pewresearch.org 24 21 33 45 22 18 19 15 May 2018 Aug 2017 66 33 57 41 19 16 26 25 54 58 May 2018'
  - page 3: top `text_span` weight=1.0, raw=3.055, weighted=3.055, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='2 PEW RESEARCH CENTER www.pewresearch.org Poor Not good Good Excellent 36 7 61 9 19 2 Total Rep/Lean Rep Dem/Lean Dem 39'
  - page 10: top `text_span` weight=1.0, raw=4.177, weighted=4.177, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='9 PEW RESEARCH CENTER www.pewresearch.org 50 41 48 50 42 42 Republican Party Democratic Party Has good policy ideas Has '
- branch `text-top-10-question` original [3, 7, 8, 2] -> original_plus [2, 3, 7, 8]
  - page 3: top `text_span` weight=1.0, raw=2.672, weighted=2.672, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='2 PEW RESEARCH CENTER www.pewresearch.org Poor Not good Good Excellent 36 7 61 9 19 2 Total Rep/Lean Rep Dem/Lean Dem 39'
  - page 2: top `text_span` weight=1.0, raw=5.570, weighted=5.570, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='PEW RESEARCH CENTER www.pewresearch.org 24 21 33 45 22 18 19 15 May 2018 Aug 2017 66 33 57 41 19 16 26 25 54 58 May 2018'

## Record 211
Doc: `05-03-18-political-release.pdf`
Question: What percentage of the party that holds the highest total percentage of good policy ideas and high ethical standards reported that they followed the news about congressional elections in their state very closely in the survey of U.S. adults conducted from April 25 to May 1, 2022?
Original answer: The survey data is from 2018, not 2022. Based on the available data, the Democratic Party has the highest total percentage (50% for good policy ideas + 42% for high ethical standards = 92%) compared to the Republican Party (50% + 41% = 91%). However, the specific percentage of Democratic Party members who followed congressional election news very closely is not provided in the materials. Therefore, while the Democratic Party has the highest total percentage for the two metrics, the question about news-following behavior cannot be fully answered with the given data.
Original+artifact answer: The Democratic Party has the highest total percentage (92%) for good policy ideas and high ethical standards, and 18% of Democrats followed the news about congressional elections very closely.
- branch `image-top-10-question` original [10, 2, 12, 3] -> original_plus [10, 2, 3, 12]
  - page 10: top `text_span` weight=1.0, raw=15.415, weighted=15.415, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='9 PEW RESEARCH CENTER www.pewresearch.org 50 41 48 50 42 42 Republican Party Democratic Party Has good policy ideas Has '
  - page 2: top `text_span` weight=1.0, raw=6.356, weighted=6.356, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='PEW RESEARCH CENTER www.pewresearch.org 24 21 33 45 22 18 19 15 May 2018 Aug 2017 66 33 57 41 19 16 26 25 54 58 May 2018'
  - page 3: top `text_span` weight=1.0, raw=4.452, weighted=4.452, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='2 PEW RESEARCH CENTER www.pewresearch.org Poor Not good Good Excellent 36 7 61 9 19 2 Total Rep/Lean Rep Dem/Lean Dem 39'
- branch `text-top-10-question` original [12, 7, 3, 11] -> original_plus [10, 2, 3, 12]
  - page 3: top `text_span` weight=1.0, raw=4.452, weighted=4.452, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='2 PEW RESEARCH CENTER www.pewresearch.org Poor Not good Good Excellent 36 7 61 9 19 2 Total Rep/Lean Rep Dem/Lean Dem 39'
  - page 10: top `text_span` weight=1.0, raw=15.415, weighted=15.415, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='9 PEW RESEARCH CENTER www.pewresearch.org 50 41 48 50 42 42 Republican Party Democratic Party Has good policy ideas Has '
  - page 2: top `text_span` weight=1.0, raw=6.356, weighted=6.356, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='PEW RESEARCH CENTER www.pewresearch.org 24 21 33 45 22 18 19 15 May 2018 Aug 2017 66 33 57 41 19 16 26 25 54 58 May 2018'

## Record 214
Doc: `05-03-18-political-release.pdf`
Question: What is the percentage gap between the youngest age group and the oldest age group that strongly disapproves of the way the president, who started his job in 2018, was handling his job in the survey of U.S. adults conducted April 25-May 1, 2018? Please write the answer in float format
Original answer: 7.0
Original+artifact answer: 0.0
- branch `image-top-10-question` original [9, 2, 7, 5] -> original_plus [2, 9, 7, 5]
  - page 2: top `text_span` weight=1.0, raw=10.270, weighted=10.270, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='PEW RESEARCH CENTER www.pewresearch.org 24 21 33 45 22 18 19 15 May 2018 Aug 2017 66 33 57 41 19 16 26 25 54 58 May 2018'
  - page 10: top `caption` weight=1.0, raw=0.000, weighted=0.000, element_locatable=None, anchors=['text_block'], locators=['source_block', 'caption_block', 'text_offset'], content='Mock caption candidate anchored to a text block.'
  - page 3: top `caption` weight=1.0, raw=0.000, weighted=0.000, element_locatable=None, anchors=['text_block'], locators=['source_block', 'caption_block', 'text_offset'], content='Mock caption candidate anchored to a text block.'
- branch `text-top-10-question` original [9, 6, 4, 12] -> original_plus [2, 9, 6, 4]
  - page 2: top `text_span` weight=1.0, raw=6.800, weighted=6.800, element_locatable=None, anchors=['text_block'], locators=['source_block', 'text_offset'], content='PEW RESEARCH CENTER www.pewresearch.org 24 21 33 45 22 18 19 15 May 2018 Aug 2017 66 33 57 41 19 16 26 25 54 58 May 2018'
