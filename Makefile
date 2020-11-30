CMD := echo
CMD := sbatch submit.sh
CMD := python
FILE := main

# username
USR := $(shell whoami | head -c 2)

# subject id
SID := 661

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
E_LIST := $(shell seq 1 5)

# 116 - 717
# E_LIST=10 27 36 37 38 4 47 112 113 114 116 117 119 120 121 122 126 71 74 75 \
         78 79 80 86 87 88 158 174 175 176

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
NP := 5
LAGS := {-2000..2000..25}
SH := --shuffle
DT := $(shell date +"%Y%m%d")
WS := 200
GPT2 := 1
GLOVE := 1
MWF := 1  # minimum word frequency 

# submit on the cluster (one job for each electrode)
run-perm-cluster:
	for elec in $(E_LIST); do \
		$(CMD) podenc_$(FILE).py \
			--sid $(SID) \
			--window-size $(WS) \
			--datum-emb-fn $(DS) \
			--word-value $(WV) \
			--$(NW) \
			--glove $(GLOVE) \
			--gpt2 $(GPT2) \
			--electrodes $$elec \
			--npermutations $(NP) \
			--min-word-freq $(MWF) \
			--lags $(LAGS) \
			$(SH) \
			--outName $(DT)-$(USR)-$(WS)ms; \
	done

run-perm-cluster1:
	$(CMD) podenc_$(FILE).py \
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
		--sig-elec-name $(SE) \
		--min-word-freq $(MWF) \
		$(SH) \
		--outName $(DT)-$(USR)-$(WS)ms; \

# submit on the command line
run-perm-cmd0:
	for elec in $(E_LIST); do \
		$(CMD) podcast-$(FILE).py \
			--sid 661 \
			--datum-emb-fn $(DS) \
			--word-value $(WV) \
			--$(NW) \
			--glove 1 \
			--electrode $$elec \
			--lags $(LAGS) \
			--npermutations $(NP) \
			$(SH) \
			--outName $(SID)-$(USR)-test1 & \
	done


# All electrodes in one job
run-perm-cmd:
	CMD := sbatch submit.sh
	for elec in $(E_LIST); do \
		$(CMD) podcast-$(FILE).py \
			--sid 661 \
			--datum-emb-fn $(DS) \
			--word-value $(WV) \
			--$(NW) \
			--glove 1 \
			--electrode $$elec \
			--lags $(LAGS) \
			--npermutations $(NP) \
			$(SH) \
			--outName $(SID)-$(USR)-test1 & \
	done