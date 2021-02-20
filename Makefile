CMD := echo
CMD := sbatch submit.sh
CMD := python
CMD := sbatch --array=1-115 submit.sh
FILE := main

# username
USR := $(shell whoami | head -c 2)

# subject id
SID := 661
# SID := 662
# SID := 717
# SID := 723
# SID := 741
# SID := 742
# SID := 763
# SID := 798

#\
661 115 electrodes \
661 96  electrodes \
717 255 electrodes \
723 165 electrodes \
741 130 electrodes \
742 175 electrodes \
763 76  electrodes \
798 195 electrodes \
#\

# Choose electrode susbset to use
# fdr  From sig-elec-50d-FDR-allLags-allElec_updated.xlsx {1000..1158}
# fdr  From FDR on max distribution: static {2000..2078}
# fdr4 From FDR on max distribution: contextual (BERT) {3000..3156}
# 14 - sig-elec-50d-pred-allElec-allLags - {200..213}
# 22 - sig-elec-50d-pred - {300..321}
# 79 - sig-elec-50d-FDR-allLags-allElec-onethresh-updated - {700..778}
# 44 - sig-elec-bert-glove-diff-FDR.csv (fdr5) - {4000..4043}
# E = $(words $(E_LIST))
# FDR across lags; FDR across elctrodes at max correlation (fdr2)
# E_LIST := $(shell seq 100 183)
# E_LIST := $(shell seq 400 485) # electest2-intersect
# E_LIST := $(shell seq 500 577) # GLoVe 5000 (0.001 sig)
# E_LIST := $(shell seq 950 987) # GLoVe 5000: -1000 to -100ms (0.05 sig)
# E_LIST := $(shell seq 2500 2560) # bert bert50d-glove50d-diff-sig-elec-01-116-abs
# E_LIST := $(shell seq 2600 2673) # gpt2-glove-50d-previous-diff-sig-elec-01-116
# E_LIST := $(shell seq 800 915) # 116 - GLoVe 5000 (0.01 sig)
# E_LIST := $(shell seq 160 170)

# 116 - 717
# E_LIST=10 27 36 37 38 4 47 112 113 114 116 117 119 120 121 122 126 71 74 75 \
         78 79 80 86 87 88 158 174 175 176

E_LIST=$(shell seq 1 1)

# Choose which word column to use.
# Options: word lemmatized_word stemmed_word
WORD_COL = lemmatized_word
WD = lemma

# Choose which stop word column to use.
# Options: is_stopword is_nltk_stop is_onix_stop is_uncontent None
EXC_COL = None
ED = none

# predictability column
PRED_COL := bart_target_prob
PRED_COL := gpt2_xl_target_prob
PRED_COL := human_target_prob
PD = $(shell echo ${PRED_COL} | head -c 4)

# datum
DS := podcast-datum-glove-50d.csv
# DS := podcast-datum-gpt2-xl-c_1024-previous-pca_50d.csv

# SE := 5000-sig-elec-50d-onethresh-01.csv
NW := nonWords
WV := all
NP := 5000
LAGS := {-2000..2000..25}
DT := $(shell date +"%Y%m%d-%H%M")
WS := 200
GPT2 := 0
GLOVE := 0
MWF := 1
# SH := --shuffle
PSH := --phase-shuffle
# PIL := mturk


PDIR := $(shell dirname `pwd`)
link-data:
	ln -fs $(PDIR)/podcast-pickling/results/* data/


run-perm-cluster:
	mkdir -p logs
	$(CMD) code/podenc_$(FILE).py \
		--sid $(SID) \
		--electrodes $(E_LIST) \
		--datum-emb-fn $(DS) \
		--window-size $(WS) \
		--word-value $(WV) \
		--$(NW) \
		--glove $(GLOVE) \
		--gpt2 $(GPT2) \
		--npermutations $(NP) \
		--lags $(LAGS) \
		--sig-elec-file $(SE) \
		--min-word-freq $(MWF) \
		$(SH) \
		$(PSH) \
		--output-prefix $(DT)-$(USR)-$(WV)-$(PIL); \

# Array jobs
# submit on the cluster (one job for each electrode)
run-perm-array:
	mkdir -p logs
	$(CMD) code/podenc_$(FILE).py \
		--sid $(SID) \
		--datum-emb-fn $(DS) \
		--window-size $(WS) \
		--word-value $(WV) \
		--$(NW) \
		--glove $(GLOVE) \
		--gpt2 $(GPT2) \
		--npermutations $(NP) \
		--lags $(LAGS) \
		--sig-elec-file $(SE) \
		--min-word-freq $(MWF) \
		$(SH) \
		$(PSH) \
		--output-prefix $(DT)-$(USR)-$(WV)-$(PIL); \
