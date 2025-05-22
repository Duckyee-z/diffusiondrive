import glob
import argparse
from pathlib import Path
import pandas as pd
import os

def find_csv_files(folder_path):
    """查找指定文件夹及其子文件夹中的所有CSV文件"""
    csv_files = []
    folder_path = Path(folder_path)
    
    if not folder_path.is_dir():
        print(f"错误：'{folder_path}' 不是一个有效的文件夹路径")
        return csv_files
    
    try:
        csv_files = list(folder_path.glob('**/*.csv'))
    except Exception as e:
        print(f"查找CSV文件时出错: {e}")
    
    return csv_files


def process_csv_files(csv_files, output_dir):
    """处理CSV文件并提取指定数据"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义原始列名和对应的新列名映射
    column_mapping = {
        'score': 'PDMS',
        'no_at_fault_collisions': 'NC',
        'drivable_area_compliance': 'DAC',
        'time_to_collision_within_bound': 'TTC',
        'comfort': 'Comf.',
        'ego_progress': 'EP',
    }
    
    # 原始需要提取的列
    original_columns = list(column_mapping.keys())
    
    all_results = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # 查找token为average的行
            average_row = df[df['token'] == 'average']
            if not average_row.empty:
                # 提取需要的列
                extracted_data = average_row[original_columns].copy()
                # 重命名列
                extracted_data.rename(columns=column_mapping, inplace=True)
                # 添加文件名信息
                extracted_data['source_file'] = file.name
                all_results.append(extracted_data)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
    
    if all_results:
        # 合并所有结果
        result_df = pd.concat(all_results, ignore_index=True)
        max_pdms_row = result_df.loc[result_df['PDMS'].idxmax()]
        # 保存到新文件
        output_path = os.path.join(output_dir, 'extracted_results.csv')
        result_df.to_csv(output_path, index=False)

        print(max_pdms_row[['PDMS', 'NC', 'DAC', 'TTC', 'Comf.', 'EP']])
        return result_df
    else:
        print("\n没有找到任何包含token='average'的CSV文件")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='查找指定目录中的所有CSV文件')
    parser.add_argument('--directory', help='要搜索的目标目录路径')
    parser.add_argument('--output', help='输出目录路径', default='./')
    

    args = parser.parse_args()
    
    csv_files = find_csv_files(args.directory)
    

    print(f"在目录 {args.directory} 中搜索...")
    
    print(f"\n共找到 {len(csv_files)} 个CSV文件")
    for file in csv_files:
        print(file)
    process_csv_files(csv_files, args.output)
    