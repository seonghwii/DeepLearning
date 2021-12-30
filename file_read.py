def read_file(file_name):

    rtn_list = []
    cnt = 0

    #file을 열면 꼭 닫아주어야 한다. | open, close
    try:
        f = open(file_name, mode="rt")

        while True:
            line = f.readline()
            if not line: break

            #map : 리스트의 요소를 지정된 함수, 내장함수 (float으로 형변환)
            line = list(map(float, (line.rstrip('\n').split("\t"))))
            rtn_list.append(line)
            cnt += 1


        f.close()

        return rtn_list, cnt

    except FileNotFoundError as e:
            print(e)

if __name__ == '__main__':

    print(read_file("sell_house.txt"))






