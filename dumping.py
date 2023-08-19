import pickle
# Get some data from loop.
a = []
for i in range(10):
    a.append(i)

# Save the data to file with pickle.
def dump(result):
    file = open('dump_file.json', 'wb')
    pickle.dump(result, file)
    file.close()
    return 'finish dumping!'

# Read the saved data.
def read_dump(file):
    # input file directory
    file_r = open(file, 'rb')
    data = pickle.load(file_r)
    file_r.close()
    print(data)
    print('The pickle saved data is : \n')
    # for i in data:
    #     print(i)
    return 0

if __name__ == '__main__':
    print(dump(a))
    print(read_dump('dump_file.json'))