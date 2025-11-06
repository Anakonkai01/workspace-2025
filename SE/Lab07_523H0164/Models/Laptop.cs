using System.ComponentModel.DataAnnotations;

namespace QuanLyLaptop.Models
{
    public class Laptop
    {
        [Display(Name = "Mã ID")]
        public int ID { get; set; }

        [Required(ErrorMessage = "Tên laptop không được để trống")]
        [Display(Name = "Tên Laptop")]
        public string Name { get; set; }

        [Range(4, 64, ErrorMessage = "RAM phải từ 4 đến 64 GB")]
        [Display(Name = "Dung lượng RAM (GB)")]
        public int RAM { get; set; }

        [Range(100, 10000, ErrorMessage = "Giá phải từ 100 đến 10000 USD")]
        [Display(Name = "Giá (USD)")]
        public int Price { get; set; }
    }
}