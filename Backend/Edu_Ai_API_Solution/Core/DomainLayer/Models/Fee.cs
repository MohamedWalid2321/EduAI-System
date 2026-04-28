using DomainLayer.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
    public class Fee : BaseEntity<int>
    {
        public int AcademicYearId { get; set; }
        public AcademicYear AcademicYear { get; set; } = null!;

        public int DepartmentId { get; set; }
        public Department Department { get; set; } = null!;

        public FeeType FeeType { get; set; }
        public decimal Amount { get; set; }
    }
}
