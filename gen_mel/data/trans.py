
fw_en = open('en_train.txt', 'w', encoding = 'utf-8')
fw_de = open('de_train.txt', 'w', encoding = 'utf-8')

with open('train.txt', 'r', encoding = 'utf-8') as f:
    for line in f:
        field = line.strip().split('\t')
        if len(field) > 2: continue
        fw_en.write(field[0] + '\n')
        fw_de.write(field[1] + '\n')

fw_en.close()
fw_de.close()

fw_en = open('en_test.txt', 'w', encoding = 'utf-8')
fw_de = open('de_test.txt', 'w', encoding = 'utf-8')

with open('test.txt', 'r', encoding = 'utf-8') as f:
    for line in f:
        field = line.strip().split('\t')
        if len(field) > 2: continue
        fw_en.write(field[0] + '\n')
        fw_de.write(field[1] + '\n')

fw_en.close()
fw_de.close()
