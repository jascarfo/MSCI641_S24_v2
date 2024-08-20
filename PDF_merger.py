import os
from PyPDF2 import PdfMerger

# Define the path to the folder containing the PDF files
pdf_folder_path = '/mnt/c/Users/joesc/OneDrive - University of Waterloo/MES_Thesis_Data/SASB_Standards'
# Define the path and name of the output PDF file
output_pdf_path = '/mnt/c/Users/joesc/OneDrive - University of Waterloo/MES_Thesis_Data/SASB_Standards_Combined.pdf'

# Create a PdfMerger object
merger = PdfMerger()

# Initialize a counter to keep track of processed files
file_count = 0

# Loop through all files in the folder
for filename in os.listdir(pdf_folder_path):
    if filename.endswith('.pdf'):
        file_path = os.path.join(pdf_folder_path, filename)
        # Append the current PDF file to the merger
        merger.append(file_path)
        file_count += 1
        print(f"Processing file {file_count}: {filename}")

# Write out the merged PDF file
merger.write(output_pdf_path)
# Close the merger
merger.close()

print(f"All PDF files have been merged into {output_pdf_path}")

