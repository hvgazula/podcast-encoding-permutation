CMD := sbatch submit.sh
# {echo | python | sbatch submit.sh | sbatch --array=1-5 submit.sh}
FILE := main

# username
USR := $(shell whoami | head -c 2)

# subject id
SID := 661
ELIST :=  $(shell seq 1 115)
SID := 662
ELIST :=  $(shell seq 1 100)
# SID := 717
# ELIST :=  $(shell seq 1 255)
# SID := 723
# ELIST :=  $(shell seq 1 165)
# SID := 741
# ELIST :=  $(shell seq 1 130)
# SID := 742
# ELIST :=  $(shell seq 1 175)
# SID := 743
# ELIST :=  $(shell seq 1 125)
# SID := 763
# ELIST :=  $(shell seq 1 80)
# SID := 798
# ELIST :=  $(shell seq 1 195)

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
# {bart_target_prob | gpt2_xl_target_prob | human_target_prob}
PD = $(shell echo ${PRED_COL} | head -c 4)

# datum
DS := podcast-datum-glove-50d.csv
# DS := podcast-datum-gpt2-xl-c_1024-previous-pca_50d.csv

# SE := 5000-sig-elec-50d-onethresh-01.csv
FOLD_IDX := $(shell seq 0 4)
REP := --replication
NW := nonWords
WV := all
NP := 1000
LAGS := {-2000..2000..25}
DT := $(shell date +"%Y%m%d")
WS := 200
GPT2 := 1
GLOVE := 1
MWF := 1
SH := --shuffle
# PSH := --phase-shuffle
# PIL := mturk

PDIR := $(shell dirname `pwd`)
link-data:
	ln -fs $(PDIR)/podcast-pickling/results/* data/


simple-encoding:
	mkdir -p logs
	$(CMD) code/podenc_$(FILE).py \
		--sid $(SID) \
		--electrodes $(ELIST) \
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
		--output-parent-dir no-shuffle \
		--output-prefix '';\


encoding-perm-cluster:
	mkdir -p logs
	for elec in $(ELIST); do \
		# for jobid in $(shell seq 1 1); do \
			$(CMD) code/podenc_$(FILE).py \
				--sid $(SID) \
				--electrodes $$elec \
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
				--output-parent-dir 20210602-paper-review-label-shuffle \
				--output-prefix '' \
				# --job-id $$jobid; \
		# done; \
	done;


encoding-perm-cluster-rep:
	mkdir -p logs
	for fold_idx in $(FOLD_IDX); do \
		for elec in $(ELIST); do \
			# for jobid in $(shell seq 1 1); do \
				$(CMD) code/podenc_$(FILE).py \
					--sid $(SID) \
					--electrodes $$elec \
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
					--fold-idx $$fold_idx \
					$(SH) \
					$(PSH) \
					$(REP) \
					--output-parent-dir 20210602-paper-review-label-shuffle \
					--output-prefix '' \
					# --job-id $$jobid; \
			# done; \
		done; \
	done;


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
