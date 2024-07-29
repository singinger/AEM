import os

folder = './data/VisDA-2017'
domains = os.listdir(folder)
domains.sort()

# def count_files_in_subdirectories(folder_path):
#     total_files = 0
#     for a, b, files in os.walk(folder_path):
#         total_files += len(files)
#     return total_files

# # 示例用法
# folder_path = './data/office31/webcam'
# num_files = count_files_in_subdirectories(folder_path)
# print("Total number of files in subdirectories:", num_files)

for d in range(len(domains)):
	dom = domains[d]
	if os.path.isdir(os.path.join(folder, dom)):
		dom_new = dom.replace(" ","_")
		print(dom, dom_new)
		os.rename(os.path.join(folder, dom), os.path.join(folder, dom_new))

		classes = os.listdir(os.path.join(folder, dom_new))
		classes.sort()
		# print(classes)
		f = open(dom_new[0] + "_list.txt", "w")
		for c in range(len(classes)):
			cla = classes[c]
			cla_new = cla.replace(" ","_")
			print(cla, cla_new)
			os.rename(os.path.join(folder, dom_new, cla), os.path.join(folder, dom_new, cla_new))
			files = os.listdir(os.path.join(folder, dom_new, cla_new))
			files.sort()
			# print(files)
			for file in files:
				file_new = file.replace(" ","_")
				os.rename(os.path.join(folder, dom_new, cla_new, file), os.path.join(folder, dom_new, cla_new, file_new))
				print(file, file_new)
				print('{:} {:}'.format(os.path.join(folder, dom_new, cla_new, file_new), c))
				f.write('{:} {:}\n'.format(os.path.join(folder, dom_new, cla_new, file_new), c))
		f.close()

