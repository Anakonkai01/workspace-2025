using QuanLyLaptop.Models;
using System.Collections.Generic;
using System.Linq;

namespace QuanLyLaptop.DAO
{
    public class LaptopDAO
    {
        [cite_start]// Dùng static list để giả lập database 
        private static List<Laptop> laptopList = new List<Laptop>
        {
            new Laptop { ID = 1, Name = "Laptop Gaming A", RAM = 16, Price = 1500 },
            new Laptop { ID = 2, Name = "Laptop Van Phong B", RAM = 8, Price = 800 },
            new Laptop { ID = 3, Name = "Laptop Do Hoa C", RAM = 32, Price = 2500 }
        };

        [cite_start]// Phương thức lấy toàn bộ danh sách 
        public List<Laptop> GetAll()
        {
            return laptopList;
        }

        [cite_start]// Phương thức lấy chi tiết một laptop 
        public Laptop GetById(int id)
        {
            return laptopList.FirstOrDefault(l => l.ID == id);
        }

        [cite_start]// Phương thức thêm laptop mới 
        public void Add(Laptop laptop)
        {
            // Tự động gán ID tăng dần
            int newId = laptopList.Max(l => l.ID) + 1;
            laptop.ID = newId;
            laptopList.Add(laptop);
        }

        [cite_start]// Phương thức cập nhật laptop 
        public void Update(Laptop laptop)
        {
            Laptop existingLaptop = laptopList.FirstOrDefault(l => l.ID == laptop.ID);
            if (existingLaptop != null)
            {
                existingLaptop.Name = laptop.Name;
                existingLaptop.RAM = laptop.RAM;
                existingLaptop.Price = laptop.Price;
            }
        }

        [cite_start]// Phương thức xóa laptop 
        public void Delete(int id)
        {
            Laptop laptop = laptopList.FirstOrDefault(l => l.ID == id);
            if (laptop != null)
            {
                laptopList.Remove(laptop);
            }
        }
    }
}