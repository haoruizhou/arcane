# Arcane ML & Grouping Logic

This document explains how Arcane calculates quality scores and groups similar photos.

## 1. Quality Scoring (0-100 Scale)

All Machine Learning scores in Arcane are standardized to a **0-100 scale**, where:
- **0-40**: Poor / Low Confidence
- **40-70**: Acceptable
- **70-100**: Excellent / High Confidence

### Focus Score (Sharpness)
The Focus Score measures the sharpness of an image (or face) to detect blur.

- **Algorithm**: Laplacian Variance (`cv2.Laplacian`).
- **Standardization**: We use an asymptotic normalization function to map unbounded variance to 0-100.
  $$ \text{Score} = 100 \times (1 - e^{-\frac{\text{Variance}}{1000}}) $$
- **Why 1000?**: High-quality, sharp DSLR images often have variances between 800-2000. A constant of 1000 ensures that only likely sharp images score above 60-70, while blurry images (variance < 100) score very low.

### Eye Openness Score
Measures how "open" a subject's eyes are.

- **Algorithm**: Variance analysis of eye regions extracted via facial landmarks.
- **standardization**: Similar to Focus Score but with a specific constant for eye regions.
  $$ \text{Score} = 100 \times (1 - e^{-\frac{\text{Variance}}{150}}) $$
- **Interpretation**: A score > 50 indicates eyes are likely open.

## 2. Semantic Analysis (CLIP)

Arcane uses **OpenAI's CLIP (Contrastive Language-Image Pre-training)** model to understand the *content* of your photos.

- **Model**: `openai/clip-vit-base-patch32`
- **Output**: A 512-dimensional vector (embedding) for each image.
- **Capabilities**:
    - These embeddings represent the semantic meaning of the image (e.g., "a dog running", "sunset at beach").
    - Images with similar content will have embeddings that are mathematically close (high Cosine Similarity).

## 3. Photo Grouping

Arcane groups photos in two ways: **Events** (Time-based) and **Stacks** (Content-based).

### Event Grouping (Timeline)
- **Logic**: Time-gap clustering.
- **Threshold**: **60 seconds**.
- **Behavior**: If two consecutive photos are taken more than 60 seconds apart, a new "Event" starts. This separates different moments (e.g., Cake Cutting vs. First Dance).

### Similarity Stacks (Deduplication)
Within an Event, Arcane looks for "Stacks" of near-identical photos (burst mode, finding the best shot).

- **Algorithm**: Connected Components on a Similarity Graph.
- **Comparison**:
    1. **Primary**: **Cosine Similarity** of CLIP Embeddings.
       - **Threshold**: **0.85** (85% similarity).
       - This catches images that *look* the same even if pixels shifted slightly.
    2. **Fallback**: **dHash** (Difference Hash) if ML is disabled.
       - **Threshold**: Hamming Distance < 12.
- **Best Shot Selection**:
    - Inside a stack, the photo with the highest **Overall Score** (Focus + other factors) is marked as the **[BEST]** shot.
    - The second highest is marked as **[2ND]**.
