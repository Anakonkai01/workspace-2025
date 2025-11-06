using QuanLyLaptop.DAO;
using QuanLyLaptop.Models;
using System.Web.Mvc;

namespace QuanLyLaptop.Controllers
{
    [cite_start]public class LaptopController : Controller // 
    {
        private LaptopDAO dao = new LaptopDAO();

        [cite_start]// GET: Hiển thị danh sách 
        public ActionResult Index()
        {
            var model = dao.GetAll();
            return View(model);
        }

        [cite_start]// GET: Hiển thị chi tiết 
        public ActionResult Details(int id)
        {
            var model = dao.GetById(id);
            if (model == null)
            {
                return HttpNotFound();
            }
            return View(model);
        }

        [cite_start]// GET: Hiển thị form thêm mới 
        public ActionResult Create()
        {
            return View();
        }

        [cite_start]// POST: Xử lý thêm mới 
        [HttpPost]
        public ActionResult Create(Laptop laptop)
        {
            [cite_start]if (ModelState.IsValid) // Kiểm tra validation 
            {
                dao.Add(laptop);
                return RedirectToAction("Index");
            }
            // Nếu không hợp lệ, hiển thị lại form với lỗi
            return View(laptop);
        }

        [cite_start]// GET: Hiển thị form chỉnh sửa 
        public ActionResult Edit(int id)
        {
            var model = dao.GetById(id);
            if (model == null)
            {
                return HttpNotFound();
            }
            return View(model);
        }

        [cite_start]// POST: Xử lý chỉnh sửa 
        [HttpPost]
        public ActionResult Edit(Laptop laptop)
        {
            [cite_start]if (ModelState.IsValid) // Kiểm tra validation 
            {
                dao.Update(laptop);
                return RedirectToAction("Index");
            }
            return View(laptop);
        }

        [cite_start]// GET: Hiển thị form xác nhận xóa 
        public ActionResult Delete(int id)
        {
            var model = dao.GetById(id);
            if (model == null)
            {
                return HttpNotFound();
            }
            return View(model);
        }

        [cite_start]// POST: Xử lý xóa 
        [HttpPost, ActionName("Delete")]
        public ActionResult DeleteConfirmed(int id)
        {
            dao.Delete(id);
            return RedirectToAction("Index");
        }
    }
}