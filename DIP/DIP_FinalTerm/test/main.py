import numpy as np
import matplotlib.pyplot as plt

def F(x, y):
    """Hệ phương trình phi tuyến"""
    f1 = np.exp(y) * np.sin(x) + np.log(y)
    f2 = np.exp(x) * np.cos(x) + y**2 - 2
    return np.array([f1, f2])

def J(x, y):
    """Ma trận Jacobian"""
    df1_dx = np.exp(y) * np.cos(x)
    df1_dy = np.exp(y) * np.sin(x) + 1/y
    df2_dx = np.exp(x) * np.cos(x) - np.exp(x) * np.sin(x)
    df2_dy = 2 * y
    return np.array([[df1_dx, df1_dy], [df2_dx, df2_dy]])

def broyden(x0, y0, saiso=0.000001, lapmax=100):
    """Phương pháp Broyden giải hệ phương trình phi tuyến"""
    giatrix = [x0]
    giatriy = [y0]
    sai = []
    saix = []
    saiy = []
    
    x, y = x0, y0
    nghiemthat = [0, 1]
    B = J(x, y)
    
    for i in range(lapmax):
        # Tính F(x,y) và giải hệ phương trình tuyến tính
        Fx = F(x, y)
        dx = np.linalg.solve(B, -Fx)
        
        # Cập nhật nghiệm
        x_moi = x + dx[0]
        y_moi = y + dx[1]
        
        # Công thức cập nhật Broyden
        s_k = np.array([x_moi - x, y_moi - y])
        y_k = F(x_moi, y_moi) - Fx
        B = B + np.outer((y_k - np.dot(B, s_k)), s_k) / np.dot(s_k, s_k)
        
        # Cập nhật giá trị
        x = x_moi
        y = y_moi
        
        # Tính sai số
        lechx = abs(x - nghiemthat[0])
        lechy = abs(y - nghiemthat[1])
        saix.append(lechx)
        saiy.append(lechy)
        lech = np.sqrt(lechx**2 + lechy**2)
        sai.append(lech)
        
        giatrix.append(x)
        giatriy.append(y)
        
        # Kiểm tra điều kiện dừng
        if lech < saiso:
            print(f"Hội tụ sau {i+1} vòng lặp")
            break
    
    return np.array(giatrix), np.array(giatriy), np.array(sai), np.array(saix), np.array(saiy)

if __name__ == "__main__":
    # Điểm khởi tạo
    x0, y0 = 0.3, 0.9
    
    # Chạy thuật toán Broyden
    giatrix, giatriy, saiso, saisox, saisoy = broyden(x0, y0)
    
    # In kết quả
    print(f"Nghiệm cuối cùng: x = {giatrix[-1]:.10f}, y = {giatriy[-1]:.10f}")
    print(f"Sai số cuối cùng: {saiso[-1]:.2e}")
    print(f"Số vòng lặp: {len(saiso)}")
    
    # Đồ thị sai số theo số vòng lặp
    lan = np.arange(1, len(saiso) + 1)
    plt.figure(figsize=(7, 5))
    plt.semilogy(lan, saiso, 'o-', color='green', linewidth=2, markersize=6)
    plt.xlabel('Số vòng lặp', fontsize=12)
    plt.ylabel('Sai số (log scale)', fontsize=12)
    plt.title('Sai số theo số vòng lặp - Phương pháp Broyden', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(['Sai số tổng'], fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Đồ thị bậc hội tụ
    if len(saiso) > 2:
        dsk = []
        for i in range(1, len(saiso) - 1):
            etrc = saiso[i-1]
            etai = saiso[i]
            esau = saiso[i+1]
            if etrc > 0 and etai > 0:
                ki = np.log(esau/etai) / np.log(etai/etrc)
                dsk.append(ki)
        
        if dsk:
            k = round(np.mean(dsk), 4)
            print(f"Bậc hội tụ ước lượng: k = {k}")
            
            en = saiso[:-1]
            en1 = saiso[1:]
            
            plt.figure(figsize=(7, 5))
            plt.loglog(en, en1, 'o-', label=f'Thực tế k = {k}', color='green', linewidth=2, markersize=6)
            
            # Đường lý thuyết (hội tụ siêu tuyến tính)
            ye = en**1.5  # Giữa tuyến tính và bậc 2
            plt.loglog(en, ye, '--', label='Lý thuyết 1 < k < 2', color='blue', linewidth=2)
            
            plt.xlabel('$e_n$', fontsize=12)
            plt.ylabel('$e_{n+1}$', fontsize=12)
            plt.title('Đồ thị bậc hội tụ', fontsize=13)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    else:
        print("Không đủ dữ liệu để tính bậc hội tụ")