from DTClassifier import timeMemory 

tm = timeMemory()

sysInfo = tm.getSysInfo()
print(sysInfo)

file = open("systemSpecs2.txt", "w+")
file.write(sysInfo)
file.close()