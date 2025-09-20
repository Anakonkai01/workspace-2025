# Cần cài đặt các thư viện cần thiết nếu bạn chưa có
# Mở terminal và chạy các lệnh sau:
# pip install requests
# pip install beautifulsoup4

import requests
from bs4 import BeautifulSoup
import time
import csv

# ==============================================================================
# HÀM 1: CÀO DỮ LIỆU CHI TIẾT TỪ MỘT TRANG SẢN PHẨM
# Hàm này được giữ nguyên, nó sẽ được gọi cho mỗi sản phẩm tìm thấy.
# ==============================================================================
def crawl_laptop_data(product_url):
    """Hàm này cào dữ liệu chi tiết từ một URL sản phẩm cụ thể."""
    print(f"  -> Đang lấy chi tiết từ: {product_url}")
    laptop_data = {'URL': product_url}
    
    try:
        response = requests.get(product_url)
        if response.status_code != 200:
            print(f"    LỖI: Không thể truy cập URL. Bỏ qua.")
            return None

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Trích xuất thông tin
        name_element = soup.find('div', class_='box-product-name')
        laptop_data['Name'] = name_element.text.strip() if name_element else 'Not Found'
        
        price_element = soup.find('div', class_='sale-price')
        if not price_element:
            price_element = soup.find('div', class_='box-price-present')
        laptop_data['Price'] = price_element.text.strip() if price_element else 'Not Found'
        
        tech_info_container = soup.find('div', class_='cps-block-technicalInfo')
        if tech_info_container:
            spec_rows = tech_info_container.find_all('tr', class_='technical-content-item')
            for row in spec_rows:
                cells = row.find_all('td')
                if len(cells) == 2:
                    spec_label = cells[0].text.strip()
                    spec_value = cells[1].text.strip()
                    
                    if 'Loại CPU' in spec_label: laptop_data['CPU'] = spec_value
                    elif 'Dung lượng RAM' in spec_label: laptop_data['RAM'] = spec_value
                    elif 'Loại card đồ họa' in spec_label: laptop_data['GPU'] = spec_value
                    elif 'Ổ cứng' in spec_label: laptop_data['Storage'] = spec_value
                    elif 'Kích thước màn hình' in spec_label: laptop_data['Screen Size'] = spec_value
                    elif 'Độ phân giải màn hình' in spec_label: laptop_data['Screen Resolution'] = spec_value
                    elif 'Chất liệu tấm nền' in spec_label: laptop_data['Panel'] = spec_value
                    elif 'Tần số quét' in spec_label: laptop_data['Screen frequency'] = spec_value
                    elif 'Trọng lượng' in spec_label: laptop_data['Weight'] = spec_value
                        
        return laptop_data

    except Exception as e:
        print(f"    LỖI: Đã có lỗi xảy ra khi xử lý URL: {e}")
        return None

# ==============================================================================
# PHẦN CHÍNH CỦA CHƯƠNG TRÌNH
# ==============================================================================

# --- GIAI ĐOẠN 1: Lấy tất cả URL sản phẩm qua API GraphQL (THAY THẾ CHO CODE CŨ) ---
GRAPHQL_URL = 'https://api.cellphones.com.vn/v2/graphql/query'
query_string = """
query GetProductsByCateId($page: Int!) {
    products(filter: {static: {categories: ["693"], province_id: 30}}, page: $page, size: 20, sort: [{view: desc}]) {
        general { url_key }
    }
}
"""
product_links = []
current_page = 1

print("--- Bắt đầu lấy tất cả URL sản phẩm qua API (xử lý 'Xem thêm') ---")

while True:
    print(f"Đang lấy danh sách URL từ trang API số: {current_page}...")
    payload = {'query': query_string, 'variables': {'page': current_page}}
    
    response = requests.post(GRAPHQL_URL, json=payload)
    
    if response.status_code != 200:
        print("Lỗi khi gọi API, dừng chương trình.")
        break
        
    data = response.json()
    products_on_page = data.get('data', {}).get('products', [])
    
    if not products_on_page:
        print("API báo hết sản phẩm. Đã lấy tất cả URL.")
        break
        
    for product in products_on_page:
        url_key = product.get('general', {}).get('url_key')
        if url_key:
            full_url = f"https://cellphones.com.vn/{url_key}.html"
            if full_url not in product_links:
                product_links.append(full_url)
            
    current_page += 1
    time.sleep(1)

print(f"\n--- HOÀN TẤT GIAI ĐOẠN 1: Tìm thấy tổng cộng {len(product_links)} URL sản phẩm. ---\n")


# --- GIAI ĐOẠN 2: Lặp qua từng URL và cào dữ liệu chi tiết (GIỮ NGUYÊN LOGIC CỦA BẠN) ---
all_laptops_data = []
print("--- Bắt đầu GIAI ĐOẠN 2: Cào dữ liệu chi tiết từ mỗi URL ---")
for link in product_links:
    data = crawl_laptop_data(link)
    if data:
        all_laptops_data.append(data)
    
    # Dừng 1 giây để không làm quá tải server
    time.sleep(1)

# --- GIAI ĐOẠN 3: Lưu kết quả ra file CSV (GIỮ NGUYÊN LOGIC CỦA BẠN) ---
csv_column = ['Name', 'Price', 'CPU', 'RAM', 'GPU','Storage', 'Screen Size', 'Screen Resolution', 'Panel', 'Screen frequency', 'Weight', 'URL']
csv_file = 'laptops_data_cellphones_full_v2.csv'
try:
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=csv_column, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_laptops_data)
    print(f"\nTHÀNH CÔNG! Toàn bộ dữ liệu đã được lưu vào file '{csv_file}'")
except IOError:
    print(f"Lỗi: Không thể ghi dữ liệu vào file CSV")