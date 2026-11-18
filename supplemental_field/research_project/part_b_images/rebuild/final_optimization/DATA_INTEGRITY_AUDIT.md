# Data integrity audit

The recheck decoded 5610 images with 0 failures. Kaggle retains 4307 of 4317 photos in 4284 groups. All five classes appear in every required split and fold.

Detected related groups crossing rebuilt splits: 0. Groups crossing CV folds: 0. Matches across current partitions and external cohorts: 0. Observer separation was checked. File hashes, decoded-pixel hashes, available Flickr IDs, DCT hashes and thumbnails detect many copies. Crops, edits or observer aliases can evade the checks.

The legacy split had 8 related groups crossing partitions. The rebuilt internal test reuses legacy development photos. All published field cohorts have been accessed. A common unseen test for the final bundles is unavailable. The runner loads only fixed development manifests and makes no historical test predictions.

Primary training uses 3,017 Kaggle photos with 645 validation photos. Supplemental training adds 450 field photos and a separate 112 field validation photos. Field features and labels never enter the primary fit. All fold-fitted transformations use fitting rows only. LDA uses fitting labels intentionally, while K-Means centroids and vocabularies receive no flower labels.

The course PDF specifies one dataset per part. No documented instructor permission for combined primary training was found. Page 6 prohibits AI-written submitted code. This AI-assisted workspace needs a permitted, student-authored submission route confirmed by the TA.

[Class/fold counts](FINAL_DATA_SPLITS.csv) · [Overlap checks](DATA_OVERLAP.csv) · [Audit summary](DATA_AUDIT.json) · [Earlier test cohorts](../final_review/TEST_COHORTS.md)
