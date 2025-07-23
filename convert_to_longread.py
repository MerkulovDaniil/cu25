import re
import sys
import os
import glob

def convert_to_longread(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split content into YAML front matter and body
    parts = content.split('---', 2)
    if len(parts) < 3:
        print("Error: Could not find YAML front matter in the input file.", file=sys.stderr)
        return

    front_matter = parts[1]
    body = parts[2]

    # 1. Update YAML front matter
    new_format = """format:
    pdf:
        include-in-header: ../files/longread_header.tex
number-sections: true"""
    
    front_matter = re.sub(r'format:.*?(?=\n\w|---)', new_format, front_matter, flags=re.DOTALL)
    front_matter = re.sub(r'header-includes:.*', '', front_matter, flags=re.DOTALL)

    # 2. Process body
    # Remove pauses
    body = re.sub(r'\n\. \. \.\n', '\n', body)
    
    # Remove slide separators
    body = re.sub(r'\n---\n', '\n', body)

    # Handle \uncover commands
    body = re.sub(r'\\uncover<.*?>\{(.*?)\}', r'\1', body, flags=re.DOTALL)

    # Remove duplicate slide titles and clean up attributes
    lines = body.split('\n')
    processed_lines = []
    last_title = None
    for line in lines:
        # Remove duplicate titles
        if line.strip().startswith('## '):
            if line.strip() == last_title:
                continue
            last_title = line.strip()
        
        # Remove .nonincremental attribute
        line = line.replace('.nonincremental', '')
        
        processed_lines.append(line)
    
    body = '\n'.join(processed_lines)
    
    # Reassemble the file
    new_content = f"---{front_matter}---\n{body}"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_longread.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(output_dir):
        print(f"Creating output directory '{output_dir}'")
        os.makedirs(output_dir)
    elif not os.path.isdir(output_dir):
        print(f"Error: Output path '{output_dir}' exists but is not a directory.", file=sys.stderr)
        sys.exit(1)

    input_files = glob.glob(os.path.join(input_dir, '*.md'))

    if not input_files:
        print(f"No markdown files found in '{input_dir}'.")
        sys.exit(0)

    for input_path in input_files:
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        
        convert_to_longread(input_path, output_path)
        print(f"Successfully converted {input_path} to {output_path}") 