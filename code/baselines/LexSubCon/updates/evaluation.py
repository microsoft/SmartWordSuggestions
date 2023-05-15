from .generalized_average_precision import GeneralizedAveragePrecision
import subprocess
import sys
from nltk.stem.wordnet import WordNetLemmatizer


class evaluation:
    def __init__(self):
        self.comb_best = ""

    def write_results(self, filepath, change_word, id, proposed):
        f = open(filepath, "a")
        proposed_list = []
        proposed_word = ""

        for word in dict(sorted(proposed.items(), key=lambda item: item[1], reverse=True)):
            proposed_list.append(word + " " + str(proposed[word]))

        proposed_word = "\t".join(proposed_list)
        f.write("RESULT" + "\t" + change_word + " " + id + "\t" + proposed_word + "\n")
        f.close()
        return

    def write_results_list(self, filepath, change_word, id, proposed):
        f = open(filepath, "a")
        proposed_list = []
        proposed_word = ""

        for word in dict(sorted(proposed.items(), key=lambda item: item[1], reverse=True)):
            proposed_list.append(word + " " + str(proposed[word][0]))
        if len(proposed) == 0:
            proposed_word = "\t".join(proposed_list)
            f.write("RESULT" + "\t" + change_word + " " + id + "\t" + proposed_word + "\n")
            f.close()
        else:
            f.write("RESULT" + "\t" + change_word + "\n")
            f.close()
        return

    def write_results_list_temp(self, filepath, change_word, id, proposed):
        f = open(filepath, "a")
        proposed_list = []
        proposed_word = ""
        for word in proposed:
            proposed_list.append(word + " " + str(proposed[word][0]))

        proposed_word = "\t".join(proposed_list)
        f.write("RESULT" + "\t" + change_word + " " + id + "\t" + proposed_word + "\n")
        f.close()
        return

    def write_results_list(self, filepath, change_word, id, proposed):
        f = open(filepath, "a")
        proposed_list = []
        proposed_word = ""
        # for word in proposed:
        for word in dict(sorted(proposed.items(), key=lambda item: item[1], reverse=True)):
            proposed_list.append(word + " " + str(proposed[word][0]))

        proposed_word = "\t".join(proposed_list)
        f.write("RESULT" + "\t" + change_word + " " + id + "\t" + proposed_word + "\n")
        f.close()
        return

    def clean_proposed(self, proposed):
        proposed_temp = {}
        for word in proposed:
            word_temp = word.replace("_", " ")
            word_temp = word_temp.replace("-", " ")
            if word_temp not in proposed_temp:
                proposed_temp[word_temp] = proposed[word]
        return proposed_temp

    def write_results_lex_best(self, filepath, change_word, id, proposed, limit=10):
        f = open(filepath, "a")
        proposed_list = []
        proposed = self.clean_proposed(proposed)
        # for word in proposed:
        for word in dict(sorted(proposed.items(), key=lambda item: item[1], reverse=True)[:limit]):
            proposed_list.append(word)

        proposed_word = ';'.join(proposed_list)
        proposed_word = proposed_word.strip()
        id_full = change_word + " " + id

        f.write(change_word + " " + id + " :: " + proposed_word + "\n")
        f.close()
        return

    def write_results_lex_oot(self, filepath, change_word, id, proposed, limit=10, index_word=None):
        try:
            lemmatizer = WordNetLemmatizer()
            if list(proposed.keys())[0] == lemmatizer.lemmatize(change_word.split('.')[0].lower(), change_word.split('.')[1]): return 
            f = open(filepath, "a")
            proposed_list = []
            proposed = self.clean_proposed(proposed)
            # for word in proposed:
            for word in dict(sorted(proposed.items(), key=lambda item: item[1], reverse=True)[:limit]):
                proposed_list.append(word)

            proposed_word = ';'.join(proposed_list)
            proposed_word = proposed_word.strip()
            id_full = change_word + " " + id

            f.write(change_word + " " + id + ' ' + index_word +  " ::: " + proposed_word + "\n")
            f.close()
            return
        except:
            return

    def write_results_p1(self, filepath, change_word, id, proposed, limit=1):
        f = open(filepath, "a")
        proposed_list = []
        proposed_word = ""
        for word in dict(sorted(proposed.items(), key=lambda item: item[1], reverse=True)[:limit]):
            proposed_list.append(word)

        proposed_word = "\t".join(proposed_list)
        f.write("RESULT" + "\t" + change_word + " " + id + "\t" + proposed_word + "\n")
        f.close()

    def write_time(self, filepath, time):
        f = open(filepath, "w")
        f.write("TIME" + "\t" + str(time) + " sec\n")
        f.close()

    def gap_calculation(self, golden_file, output_results, results_file):
        gap_metric = GeneralizedAveragePrecision()
        gold_data = {}
        gold_file = open(golden_file, 'r', encoding="latin1")
        # ignoring words with no candidates or when they have multiple words as subtitution
        ignore_mwe = True
        for gold_line in gold_file:
            gold_instance_id, gold_weights = gap_metric.read_gold_line(gold_line, ignore_mwe)
            gold_data[gold_instance_id] = gold_weights
        ignored = 0

        eval_data = {}
        i = 0
        sum_gap = 0.0
        eval_file = open(output_results, 'r', encoding="latin1")
        for eval_line in eval_file:
            eval_instance_id, eval_weights = gap_metric.read_eval_line(eval_line)
            eval_data[eval_instance_id] = eval_weights

        ignored = 0
        out_file = open(results_file, 'w')
        randomize = False
        # how to go over the evaluation results
        for gold_instance_id, gold_weights in gold_data.items():
            try:
                eval_weights = eval_data[gold_instance_id]
            except:

                print(gold_instance_id)
                continue
            gap = GeneralizedAveragePrecision.calc(gold_weights, eval_weights, randomize)
            if (gap < 0):
                # this happens when there is nothing left to rank after filtering the multi-word expressions
                ignored += 1
                continue
            out_file.write(gold_instance_id + "\t" + str(gap) + "\n")
            i += 1
            sum_gap += gap

        mean_gap = sum_gap / i
        out_file.write("\ngold_data %d eval_data %d\n" % (len(gold_data), len(eval_data)))
        out_file.write("\nRead %d test instances\n" % i)
        out_file.write("\nIgnored %d test instances (couldn't compute gap)\n" % ignored)
        out_file.write("\nMEAN_GAP\t" + str(mean_gap) + "\n")

        gold_file.close()
        eval_file.close()
        out_file.close()

    def gap_calculation_test1(self, golden_file, output_results, results_file, combination, max_iter):
        gap_metric = GeneralizedAveragePrecision()
        gold_data = {}
        gold_file = open(golden_file, 'r')
        # ignoring words with no candidates or when they have multiple words as subtitution
        ignore_mwe = True
        for gold_line in gold_file:
            gold_instance_id, gold_weights = gap_metric.read_gold_line(gold_line, ignore_mwe)
            gold_data[gold_instance_id] = gold_weights
        ignored = 0

        eval_data = {}
        i = 0
        sum_gap = 0.0
        eval_file = open(output_results, 'r')
        for eval_line in eval_file:
            eval_instance_id, eval_weights = gap_metric.read_eval_line(eval_line)
            eval_data[eval_instance_id] = eval_weights

        ignored = 0
        randomize = False
        # how to go over the evaluation results
        for gold_instance_id, gold_weights in gold_data.items():
            eval_weights = eval_data[gold_instance_id]
            gap = GeneralizedAveragePrecision.calc(gold_weights, eval_weights, randomize)
            if (gap < 0):
                # this happens when there is nothing left to rank after filtering the multi-word expressions
                ignored += 1
                continue
            i += 1
            sum_gap += gap

        mean_gap = sum_gap / i
        gold_file.close()
        eval_file.close()
        if max_iter < mean_gap:
            max_iter = mean_gap
            self.comb_best = combination
        return max_iter

    def calculation_perl(self, golden_file, output_results_best, output_results_out, results_file_best,
                         results_file_out):
        command = "perl metrics/score.pl " + output_results_best + " " + golden_file + " -t best > " + results_file_best  # + " -v"

        subprocess.run(command, shell=True)

        command = "perl metrics/score.pl " + output_results_out + " " + golden_file + " -t oot > " + results_file_out  # + " -v"
        subprocess.run(command, shell=True)
        return

    def calculation_perl_test1(self, golden_file, output_results_best, output_results_out, results_file_best,
                               results_file_out, combination, max_iter):
        command = "perl metrics/score.pl " + output_results_best + " " + golden_file + " -t best > " + results_file_best
        subprocess.run(command, shell=True)

        gold_file = open(results_file_best, 'r')

        command = "perl metrics/score.pl " + output_results_out + " " + golden_file + " -t oot > " + results_file_out
        subprocess.run(command, shell=True)

        i = 0
        for line in gold_file:
            i = i + 1
            if i == 16:
                recall = float(line.split("=")[2].replace("\n", ""))
                precision = float(line.split("=")[1].split(",")[0])
            if i == 18:
                mode_recall = float(line.split("=")[2].replace("\n", ""))
                mode_precision = float(line.split("=")[1].split(",")[0])

        if max_iter < mode_precision:
            max_iter = mode_precision
            self.comb_best = combination

        return max_iter

    def calculation_p1(self, golden_file, output_results, results_file):
        gold_data = {}
        gold_file = open(golden_file, 'r', encoding="latin1")

        ignore_mwe = True
        for gold_line in gold_file:
            gold_instance_id, candidates_list = self.read_gold_line(gold_line, ignore_mwe)
            gold_data[gold_instance_id] = {}
            for candidate_name in candidates_list:
                gold_data[gold_instance_id][candidate_name] = 0

        eval_data = {}
        i = 0
        eval_file = open(output_results, 'r', encoding="latin1")
        for eval_line in eval_file:
            eval_instance_id, candidate_name = self.read_eval_line(eval_line)

            if candidate_name == "":
                # no candidates
                pass
            else:
                eval_data[eval_instance_id] = candidate_name

        ignored = 0
        correct = 0
        out_file = open(results_file, 'w')
        randomize = False
        # how to go over the evaluation results
        for gold_instance_id in gold_data:
            # the comment is for words that do not have an appropriate candidate
            if gold_instance_id in eval_data:
                if eval_data[gold_instance_id] in gold_data[gold_instance_id]:
                    correct = correct + 1
            else:
                ignored = ignored + 1

        p1 = correct / len(gold_data)
        out_file.write("\ngold_data %d eval_data %d\n" % (len(gold_data), len(eval_data)))
        out_file.write("\nIgnored %d test instances (couldn't be found in eval data )\n" % ignored)
        out_file.write("\nPrecision@1\t" + str(p1) + "\n")

        gold_file.close()
        eval_file.close()
        out_file.close()

    def read_gold_line(self, gold_line, ignore_mwe):
        segments = gold_line.split("::")
        instance_id = segments[0].strip()
        gold_cand = []
        line_candidates = segments[1].strip().split(';')
        for candidate_count in line_candidates:
            if len(candidate_count) > 0:
                delimiter_ind = candidate_count.rfind(' ')
                candidate = candidate_count[:delimiter_ind]
                if ignore_mwe and ((len(candidate.split(' ')) > 1) or (len(candidate.split('-')) > 1)):
                    continue
                count = candidate_count[delimiter_ind:]
                try:
                    gold_cand.append(candidate)
                except ValueError as e:
                    print(gold_line)
                    sys.exit(1)
        return instance_id, gold_cand

    def read_eval_line(self, eval_line, ignore_mwe=True):
        # get the fist candidate for p@1
        eval_cand = []
        segments = eval_line.split("\t")
        instance_id = segments[1].strip()
        for candidate in segments[2:]:
            candidate = candidate.replace("\n", "")
            if len(candidate) > 0:
                try:
                    eval_cand.append(candidate)
                    break
                except:
                    print("Error appending: %s %s" % (candidate))
        try:
            return instance_id, eval_cand[0]
        except:
            return instance_id, ""

    def write_not_found_words(self, filepath, not_found):
        f = open(filepath, "w")
        for main_word in not_found:
            for word in not_found[main_word]:
                f.write(main_word + " : " + word + "\n")
        f.close()
        return
