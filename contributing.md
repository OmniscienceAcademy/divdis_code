python run_experiment.py --data '/Users/raph/Desktop/Perso/effisc/happyFaces_utils/HappyFaces' --loss_type 1 --smooth --labeled_scale 1 --distinct_scale 1 --class_certainty_scale 1 --lr 0.0004 --model 'PretrainedResnetClassifier(18,2)' --epochs 20 --batch_size 75 --mix_rates -1 0.3

❯ ls /Users/raph/Desktop/Perso/effisc/happyFaces/HappyFaces
labeled -> diag
unlabeled -> diag and cross
test -> diag and cross
val -> diag and cross
metadata.json


❯ ls /Users/raph/Desktop/Perso/effisc/happyFaces/HappyFaces/labeled
cross diag
❯ ls /Users/raph/Desktop/Perso/effisc/happyFaces/HappyFaces/labeled/cross
❯ ls /Users/raph/Desktop/Perso/effisc/happyFaces/HappyFaces/labeled/diag
FHWH FSWS
❯ ls /Users/raph/Desktop/Perso/effisc/happyFaces/HappyFaces/labeled/diag/FHWH
James_McPherson_0001_FHWH.png       Jodie_Foster_0001_FHWH.png          ...




