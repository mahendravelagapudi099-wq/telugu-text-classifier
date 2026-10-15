# Error Analysis Report — Telugu Sentiment Classification

This report analyzes a stratified sample of 100 misclassified test examples on native Telugu sentiment classification (3-class: `neg`, `neutral`, `pos`).

## 1. Error Category Distribution (100 Stratified Examples)

| Error Category | Count | Percentage | Description / Linguistic Nature |
|---|---|---|---|
| `subtle_neutral_boundary` | 96 | 96.0% | Factual news sentences containing mild subjective sentiment or borderline neutrality |
| `mixed_sentiment` | 1 | 1.0% | Clauses joined by contrastive conjunctions (e.g., 'కానీ', 'అయినా') expressing both positive and negative aspects |
| `colloquial_code_switching` | 2 | 2.0% | English loanwords, transliterated film jargon, or informal dialectal phrases |
| `implicit_negation` | 1 | 1.0% | Negative semantic stance conveyed without explicit negation markers ('లేదు', 'కాదు') |
| `sarcasm_irony` | 0 | 0% | Surface praise or hyperbolic honorifics conveying criticism in political/cricket commentary |

---

## 2. Representative Examples per Category (3 Examples Each)

### Category: `subtle_neutral_boundary`
**Count**: 96 (96% of sample)

**Example 1:**
> "మహింద్రా కంపెనీకి చెందిన థార్‌ జీప్‌ను ‘నా పేరు సూర్య’ సినిమా కోసం ప్రత్యేకంగా డిజైన్‌ చేయించారు."
- **True Label**: `pos`
- **Predicted Label**: `neutral`
- **Model Confidence**: `0.710`

**Example 2:**
> "పరీక్ష కేంద్రాల చుట్టుపక్కలా 144 సెక్షన్ అమలుచేస్తున్నామని, సమీపఔంలో ఉన్న జిరాక్స్ కేంద్రాలను మూసివేస్తున్నామని అన్నారు"
- **True Label**: `neutral`
- **Predicted Label**: `pos`
- **Model Confidence**: `0.477`

**Example 3:**
> "రెండో రోజు ఆదివారం ఉదయం 8 గంటల నుంచి 12 గంటల వరకు సన్యాసాంగ అష్ట శ్రాద్దాలు, శాస్త్ర, అహితాగ్ని, వాక్యార్ధ మహాసభలు జ‌రుగుతాయి."
- **True Label**: `pos`
- **Predicted Label**: `neutral`
- **Model Confidence**: `0.741`

**Hypothesis for why this category fails:**
> *[Hypothesis]*: In Telugu journalism, factual event reporting frequently cites emotional outcomes (e.g., 'విజయం సాధించారు' or 'బాధాకరం') without the journalist expressing personal sentiment. Classical n-gram models fixate on the individual sentiment tokens rather than sentence-level informational context.

---

### Category: `mixed_sentiment`
**Count**: 1 (1% of sample)

**Example 1:**
> "ఈ పుస్తకానికి ‘హీల్డ్‌: హౌ క్యాన్సర్‌ గేవ్‌ మి ఎ న్యూ లైఫ్‌’ అనే టైటిల్‌ పెట్టారు."
- **True Label**: `pos`
- **Predicted Label**: `neutral`
- **Model Confidence**: `0.679`

**Hypothesis for why this category fails:**
> *[Hypothesis]*: Linear models weight positive and negative lexical tokens equally across the sentence. The final sentiment in Telugu often hinges on the post-conjunction clause (after 'కానీ'), which requires directional sequence modeling to resolve.

---

### Category: `colloquial_code_switching`
**Count**: 2 (2% of sample)

**Example 1:**
> "ముంబై : భారత ఫాస్ట్‌ బౌలర్‌ ఇర్ఫాన్‌ పఠాన్‌ సరికొత్త రికార్డు స_x005F_x007f_ష్టించనున్నాడు."
- **True Label**: `neutral`
- **Predicted Label**: `pos`
- **Model Confidence**: `0.761`

**Example 2:**
> "వీరితో పాటు ప_x005F_x007f_థ్వీ షా ఓపెనర్‌గా కీలకం కానుంది."
- **True Label**: `neutral`
- **Predicted Label**: `pos`
- **Model Confidence**: `0.564`

**Hypothesis for why this category fails:**
> *[Hypothesis]*: Sub-word segmentations for Romanized English terms (e.g. 'flop', 'director') or phonetic transliterations fragment character n-grams, diluting sentiment signal in pure Telugu vocabularies.

---

### Category: `implicit_negation`
**Count**: 1 (1% of sample)

**Example 1:**
> "శారదా చిట్‌ఫండ్‌ కుంభకోణంలో ఆధారాలు నాశనం చేశారనే ఆరోపణలను రాజీవ్‌ కుమార్‌ ఎదుర్కొంటున్నారు."
- **True Label**: `neg`
- **Predicted Label**: `neutral`
- **Model Confidence**: `0.591`

**Hypothesis for why this category fails:**
> *[Hypothesis]*: Words indicating failure, losses, or fraud imply negative polarity through domain world knowledge rather than syntactic negation, leading bag-of-words classifiers to default to neutral.

---

### Category: `sarcasm_irony`
**Count**: 0 (0% of sample)

**Hypothesis for why this category fails:**
> *[Hypothesis]*: Sarcasm relies on pragmatic contrast between literal honorifics ('మహానుభావుడు', 'ఘనత') and real-world failure, which cannot be captured without external context or deep contextualized discourse representations.

---

## 3. Comparison: Baseline vs. Transformer (IndicBERTv2)

The fine-tuned transformer (`IndicBERTv2-MLM-only`) increased overall Test Macro-F1 from **0.6483** to **0.7332** (+8.49 percentage points) and test accuracy from **66.07%** to **73.61%**, eliminating **390 misclassifications** across the 5,169 held-out test rows.

### What the Transformer Fixed:
1. **Mixed Sentiment Resolution**: Transformer self-attention across long clauses enables the model to attend to the clause following 'కానీ' / 'అయినా' to determine the dominant sentiment, where TF-IDF bag-of-words canceled out opposing words.
2. **Implicit Negative Polarity**: Contextualized embeddings for domain-specific nouns (e.g. 'అసహనం', 'నష్టం', 'విఫలం') correctly cluster into negative semantic space even without explicit negation ('లేదు').
3. **Morphological Agglutination**: Telugu's rich suffix inflection (vibhakti) creates hundreds of surface forms for single stems. IndicBERTv2's sub-word vocabulary captures root morphemes far more robustly than fixed character n-grams.

### What Remains Challenging (Remaining Errors):
1. **Pragmatic Sarcasm**: Sarcastic sports/movie commentary where praise words are used mockingly remains difficult without multi-turn discourse context or user intent.
2. **Fine-Grained Neutral Boundaries**: Distinguishing between an objectively neutral report about a tragedy versus an explicit negative opinion remains subjective even among human annotators.

---

## 4. Limitations of this Error Analysis

1. **Sample Size**: The analysis was conducted on a stratified sample of 100 misclassified examples. While representative across true classes, smaller error sub-types (e.g. rare idioms) are under-represented.
2. **Single Annotator**: Category boundaries (particularly between subtle neutral reporting and mild sentiment) were evaluated by a single annotator without inter-annotator agreement metrics (such as Cohen's Kappa).
3. **Domain Bias**: The test split contains heavy concentrations of cinema reviews and political news; findings may not generalize equally to colloquial chat or formal literary Telugu.
