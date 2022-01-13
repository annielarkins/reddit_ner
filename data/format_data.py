from curses.ascii import isspace
import jsonlines
import json

def jsonl_to_txt(jsonl_file):

    # Set up txt files
    train_txt_file = open("/Users/annielarkins/Desktop/reddit_ner/data/train/train_data.txt", "w")
    test_txt_file = open("/Users/annielarkins/Desktop/reddit_ner/data/test/test_data.txt", "w")
    with jsonlines.open(jsonl_file) as f:
        line_num = 0
        for line in f.iter():
            # Determine which file to write to
            if line_num % 3 == 0:
                txt_file = test_txt_file
            else:
                txt_file = train_txt_file
            line_num += 1
            
            # Parse
            for token in line[u'tokens']:
                current_word = str(token[u'text'].encode('utf-8'))
                # Check if label exists
                if u'label' in token:
                    current_label = str(token[u'label'].encode('utf-8'))
                else:
                    current_label = "NA"
                if str.isspace(current_word):
                    print(line_num)
                    print("HERE---" + current_word + "---")
                else:
                    txt_file.write(current_word + "\t" + current_label + "\n")
            txt_file.write("\n")

    train_txt_file.close()
    test_txt_file.close()

# TODO
def txt_to_jsonl(txt_file):
    pass

jsonl_to_txt('/Users/annielarkins/Desktop/reddit_ner/data/reddit-jargon.jsonl')