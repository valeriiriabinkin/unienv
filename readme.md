import os
import pandas as pd

def find_excel_files(directory):
    """Находит все Excel файлы в указанной директории и поддиректориях."""
    excel_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".xls"):
                excel_files.append(os.path.join(root, file))
    return excel_files

def combine_excel_files(directory, output_file):
    """Объединяет все Excel файлы из указанной директории и сохраняет в один файл."""
    excel_files = find_excel_files(directory)
    combined_df = pd.DataFrame()
    
    for file in excel_files:
        df = pd.read_excel(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    combined_df.to_excel(output_file, index=False)

# Задать директорию, где находятся папки с Excel файлами, и имя выходного файла
directory = 'path_to_your_directory'
output_file = 'combined_excel_file.xlsx'

# Выполнить объединение файлов
combine_excel_files(directory, output_file)
