import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize

if __name__ == '__main__':

    sumInputSec, sumOutputSec = [], []
    with open("./section_titles/mimic_sec_SUM.txt", "r", encoding="UTF-8") as f_read:
        sumSec = f_read.readlines()
        sumSec = [_.strip().lower() for _ in sumSec if len(_.strip()) > 0]
        special_index = sumSec.index('===============')
        
        sumOutputSec = list(sumSec[:special_index])
        sumInputSec = list(sumSec[special_index+1:])

    print(sumInputSec)
    print(sumOutputSec)
    sumSections = [_ for _ in sumInputSec]
    sumSections.extend(sumOutputSec)

    chunksize = 50000
    target_path = "../mimic-iv-note-2.2/note"
    csv_name = "discharge.csv"
    current_target = "mimic-iv-" + csv_name.split(".")[0]

    avg_sent_input, avg_word_input = 0, 0
    avg_sent, avg_word = 0, 0
    reports_list = []
    cur_idx = 0
    with pd.read_csv(target_path + csv_name, chunksize=chunksize) as reader:
        for chunk in reader:
            check_subject = chunk["text"]
            for report_obj in tqdm(check_subject):
                report = report_obj.lower()
                if 'final diagnoses:' not in report and 'discharge diagnosis:' not in report and 'discharge diagnoses:' not in report:
                    pass
                else:
                    search_res_list = []
                    for section in sumSections:
                        search_res = re.search(section, report)
                        if search_res is not None:
                            indice_0, indice_1 = search_res.span()
                            search_res_list.append([indice_0, indice_1, section])
                            
                    search_res_list = sorted(search_res_list, key=lambda x:x[0])
                    sumInputText, sumOutputText = '', ''
                    section_count = len(search_res_list)
                    for sec_i in range(section_count):
                        sec_tuple = search_res_list[sec_i]
                        section_title = sec_tuple[2]
                        
                        if sec_i == section_count - 1:
                            if section_title in sumInputSec:
                                sumInputText += report[sec_tuple[0]:].strip() + '\n'
                            elif section_title in sumOutputSec:
                                sumOutputText += report[sec_tuple[1]:].strip() + '\n'
                            break
                        
                        next_tuple = search_res_list[sec_i+1]
                        if section_title in sumInputSec:
                            sumInputText += report[sec_tuple[0]:next_tuple[0]].strip() + '\n'
                        elif section_title in sumOutputSec:
                            sumOutputText += report[sec_tuple[1]:next_tuple[0]].strip() + '\n'
                            
                    sumInputText = sumInputText.strip()
                    sumOutputText = sumOutputText.strip()
                    
                    sumOutputTextClean = ''
                    for sent in sumOutputText.split('\n'):
                        if re.search('last name', sent) is not None and re.search('first name', sent) is not None:
                            break
                        else:
                            sumOutputTextClean += sent + '\n'
                            
                    sumOutputText = sumOutputTextClean.strip()
                    if len(sumOutputText) > 0 and len(sumInputText) > 0:
                        reports_list.append(
                            {'source': sumInputText,
                            'summary': sumOutputText}
                        )
                        token_sent_input = sent_tokenize(sumInputText)
                        token_word_input = word_tokenize(sumInputText)
                        avg_sent_input += len(token_sent_input)
                        avg_word_input += len(token_word_input)
                        
                        token_sent = sent_tokenize(sumOutputText)
                        token_word = word_tokenize(sumOutputText)
                        
                        avg_sent += len(token_sent)
                        avg_word += len(token_word)

    reports_list_size = len(reports_list)
    print(f"reports_list, {reports_list_size}")
    print('avg_sent_input', avg_sent_input / reports_list_size, 'avg_word_input', avg_word_input / reports_list_size)
    print('avg_sent', avg_sent / reports_list_size, 'avg_word', avg_word / reports_list_size)
    to_write_json = {'data': reports_list}

    with open(os.path.join('./dataset', current_target + ".json"), 'w', encoding='utf-8') as write_f:
        write_f.write(json.dumps(to_write_json))
    reports_list = None
    to_write_json = None


    radioSec = []
    radioInputSec, radioOutputSec = [], []
    avg_sent_input, avg_word_input = 0, 0
    avg_sent, avg_word, count_text = 0, 0, 0
    section_titles = []
    with open("./section_titles/mimic_sec_RADIOLOGY.txt", "r", encoding="UTF-8") as f_read:
        radioSec = f_read.readlines()
        radioSec = [_.strip().lower() for _ in radioSec if len(_.strip()) > 0]
        special_index = radioSec.index('===============')
        
        radioInputSec = list(radioSec[:special_index])
        radioOutputSec = list(radioSec[special_index+1:])

    data_list = []

    chunksize = 50000
    csv_name = "radiology.csv"
    current_target = "mimic-iv-" + csv_name.split(".")[0]

    with pd.read_csv(target_path + csv_name, chunksize=chunksize) as reader:
        for chunk in reader:
            check_subject = chunk["text"]
            for report_obj in tqdm(check_subject):
                raw_text = report_obj.lower()

                inputText, outputText = '', ''

                search_res_list = []
                for section in radioSec:
                    search_res = re.search(section, raw_text)
                    if search_res is not None:
                        indice_0, indice_1 = search_res.span()
                        search_res_list.append([indice_0, indice_1, section])

                section_count = len(search_res_list)
                if section_count > 0:
                    search_res_list = sorted(search_res_list, key=lambda x:x[0])
                    flag1, flag2 = False, False
                    for sec_res in search_res_list:
                        if sec_res[-1] in radioInputSec:
                            flag1 = True
                        elif sec_res[-1] in radioOutputSec:
                            flag2 = True
                    if flag1 and flag2:            
                        for res_i in range(section_count):
                            sec_res = search_res_list[res_i]
                            if res_i == section_count - 1:
                                if sec_res[-1] in radioInputSec:
                                    inputText += raw_text[sec_res[0]:].strip()
                                elif sec_res[-1] in radioOutputSec:
                                    outputText += raw_text[sec_res[0]:].strip()
                            else:
                                sec_res_next = search_res_list[res_i+1]
                                
                                if sec_res[-1] in radioInputSec:
                                    inputText += raw_text[sec_res[0]:sec_res_next[0]].strip() + '\n'
                                elif sec_res[-1] in radioOutputSec:
                                    outputText += raw_text[sec_res[1]:sec_res_next[0]].strip() + '\n'
                            
                        inputText = inputText.strip()
                        outputText = outputText.strip()

                        if '______________________________________________________________________________' in outputText:
                            clean_idx = outputText.index('______________________________________________________________________________')
                            outputText = outputText[:clean_idx]

                        if len(inputText) > 0 and len(outputText) > 0:
                            # write source and summary into dict
                            data_list.append(
                                {'source': inputText,
                                'summary': outputText}
                            )

                            token_sent_input = sent_tokenize(inputText)
                            token_word_input = word_tokenize(inputText)
                            avg_sent_input += len(token_sent_input)
                            avg_word_input += len(token_word_input)

                            token_sent = sent_tokenize(outputText)
                            token_word = word_tokenize(outputText)

                            avg_sent += len(token_sent)
                            avg_word += len(token_word)
                            count_text += 1
                        
    print('RADIOLOGY', count_text)
    print('avg_sent_input', avg_sent_input / count_text, 'avg_word_input', avg_word_input / count_text)
    print('avg_sent', avg_sent / count_text, 'avg_word', avg_word / count_text)

    to_write_json = {'data': data_list}
    with open(os.path.join('./dataset', current_target + ".json"), 'w', encoding='utf-8') as write_f:
        write_f.write(json.dumps(to_write_json))

