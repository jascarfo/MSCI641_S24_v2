# ESG Data Scraper

import os
import requests
import time

# List of tickers
tickers = [
     "RY", "TD", "SHOP", "CNQ", "TRI", "ENB", "CP", "CNR", "BN", "CSU", "BMO", "ATD", "BNS", "NGT", "SU",
    "MFC", "WCN", "CM", "TRP", "L", "IMO", "CVE", "AEM", "QSP.UN", "ABX", "IFC", "BCE", "SLF", "FFH", "NA",
    "GWO", "DOL", "WPM", "TECK.A", "TECK.B", "NTR", "FNV", "GIB.A", "T", "QSR", "PPL", "CCO", "WN", "RCI.A",
    "RCI.B", "WSP", "FTS", "POW", "H", "IVN", "TOU", "BAM", "GFL", "RBA", "BIP.UN", "MRU", "OVV", "TFII",
    "MG", "FM", "ARX", "K", "TPX.B", "STN", "SAP", "EMA", "CCL.A", "CCL.B", "LUN", "DSG", "OTEX", "DAY",
    "X", "PAAS", "ATRL", "TIH", "BEP.UN", "EFN", "PHYS", "CLS", "FSV", "ALA", "AGI", "IGM", "GIL", "KEY",
    "EMP.A", "BNRE", "BBD.A", "BBD.B", "CAE", "IAG", "WFG", "CTC.A", "MEG", "CTC", "CAR.UN", "CS", "CIGI",
    "EDV", "ONEX", "BLCO", "CPG", "BEPC", "PSLV", "QBR.B", "QBR.A", "U.UN", "PKI", "CEF", "PSK", "DOO",
    "AQN", "AC", "NVEI", "BIPC", "NPI", "CU", "WCP", "FTT", "ERF", "BYD", "HBM", "LUG", "DFY", "CPX",
    "REI.UN", "GLXY", "SJ", "NXE", "BTO", "PRMW", "POU", "KXS", "ELD", "ATZ", "ACO.X", "MX", "ACO.Y",
    "ATS", "GRT.UN", "CHP.UN", "TFPM", "CWB", "PBH", "OR", "LNR", "CURA", "BEI.UN", "BTE", "ELF", "GEI",
    "BHC", "EQB", "CGG", "CSH.UN", "TPZ", "BLX", "DIR.UN", "FIL", "SRU.UN", "IGAF", "FCR.UN", "ERO",
    "CRT.UN", "CIA", "EQX", "PRU", "TOY", "GSY", "LSPD", "ATH", "SES", "TA", "NVA", "PEY", "FR", "WPK",
    "MFI", "IMG", "DML", "IPCO", "OGC", "CEE", "VET", "AIF", "HR.UN", "NGD", "CRR.UN", "AQA", "XTC",
    "SYZ", "QST", "PHS.U", "PIA", "PZA", "PSI", "PHO", "PBG", "PL", "PPR", "PLZ.UN", "PTM", "PMT", "PONY",
    "PYR", "QUIS", "RCH", "RMX", "RBA", "RBA.WS", "RTG", "RVX", "ROXG", "RDL", "RME", "RMP", "RUS",
    "RZL", "S", "SBB", "SIS", "SBOT", "SDL", "SCL", "SCC", "SDE", "SII", "SSRM", "SMP", "SOY", "SOLO",
    "SCCB", "STEP", "STP", "SVM", "SWA", "SWE", "SYZ", "T", "TH", "TGL", "TNT.UN", "TOG", "TVK", "TVE",
    "TQ", "TXG", "TV", "TWM", "TSK", "U", "USU", "URU", "VNP", "VIT", "VLE", "VGM", "WCE", "WFG", "WFT",
    "WLK", "WLL", "WM", "WTE", "WRN", "WZR", "XCT", "YGR", "ZON"
]

years = range(2005, 2025)

# Base URL format
url_format = "https://www.responsibilityreports.com/HostedData/ResponsibilityReportArchive/{first_letter}/TSX_{ticker}_{year}.pdf"

# Directory to save the downloaded PDFs
output_dir = "C:\\Users\\joesc\\OneDrive - University of Waterloo\\MES_Thesis_Data\\TSX_ESG"
os.makedirs(output_dir, exist_ok=True)

# Function to download a PDF with rate limiting and retry logic
def download_pdf(ticker, year, retries=3, delay=1):
    first_letter = ticker[0].lower()
    url = url_format.format(first_letter=first_letter, ticker=ticker, year=year)
    
    for attempt in range(retries):
        try:
            response = requests.head(url)
            if response.status_code == 200:
                response = requests.get(url)
                file_path = os.path.join(output_dir, f"{ticker}_ESG_{year}.pdf")
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                print(f"Downloaded: {file_path}")
                return True
            else:
                print(f"Report not found for {ticker} in {year}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    print(f"Failed to download {url} after {retries} attempts")
    return False

# Iterate through tickers and years
for ticker in tickers:
    for year in years:
        download_pdf(ticker, year)
        time.sleep(1)  # Sleep to prevent rate limiting