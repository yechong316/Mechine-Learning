with open('adult_1.data', 'w') as f:

    with open('adult.data', 'r') as f1:
        data = f1.readlines()
        print(len(data))
        split_point = len(data) / 3
        data1 = data[:int(split_point)]
        for i in data1:
            f.write(i)
        # print(data1[0])
with open('adult_3.data', 'w') as f:

    with open('adult.data', 'r') as f1:
        data = f1.readlines()
        print(len(data))
        split_point = len(data) / 3
        data1 = data[2 * int(split_point): 3 * int(split_point)]
        for i in data1:
            f.write(i)