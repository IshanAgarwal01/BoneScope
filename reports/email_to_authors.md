# Draft email to the dataset authors

> Fill in the bracketed parts before sending. The corresponding author's name and
> address are on the paper's page under "Correspondence"
> (https://www.nature.com/articles/s41597-025-06027-z). They are left blank here on
> purpose rather than guessed.

**To:** [corresponding author email]
**Subject:** Patient-level grouping for the MDS bone marrow cell dataset (figshare 28737170)

Dear Dr. [Surname],

I am an undergraduate in Computer Science and Engineering at The Ohio State
University. My team is using your dataset, "A large dataset of bone marrow cells
in myelodysplastic syndrome for classification systems" (Scientific Data 12,
1849, 2025), for an educational machine-learning project in our university's Big
Data and Analytics Association. Thank you for making it publicly available.

To evaluate our classifier fairly, we want cells from the same patient to stay
together in either the training set or the test set, never both. As far as we can
tell, the public release does not include patient or slide identifiers. For now we
are approximating this with clustering, and we will label our results as possibly
optimistic.

Would you be able to share a mapping from image file to a de-identified patient
(or slide) code for research use? A simple anonymous integer per patient would be
enough. We would use it only for building train/test splits, would not attempt any
re-identification, would follow any data-use terms you specify, and would cite
your paper and acknowledge your help in our report.

If that isn't possible, we would also be grateful to know whether the images
within each class folder are ordered by patient, or whether any part of the
filename encodes the source slide.

Thank you for your time and for this valuable resource.

Best regards,
Ishan Agarwal
B.S. Computer Science and Engineering, The Ohio State University
[OSU email] · [optional: team members' names]
