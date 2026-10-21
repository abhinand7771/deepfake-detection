This folder is reserved for the training dataset.

The application supports labeled real and manipulated face datasets such as FaceForensics++ and compatible image datasets. The data must be obtained legally and used under the relevant license and access restrictions.

Recommended structure:

- dataset/real/  # real face images
- dataset/fake/  # manipulated face images

Do not scrape arbitrary public faces or download protected datasets without authorization.

When ready to train:
1. Place real images in dataset/real/
2. Place fake/manipulated images in dataset/fake/
3. Ensure images are readable and not corrupted.
4. Prefer split by source source video/identity when dataset metadata is available.

If no dataset is present, the app can still start and show an informative message without making fake predictions.
