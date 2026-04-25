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
        public AcademicYear AcademicYear { get; set; }

        public FeeType Name { get; set; } // Tuition / Books
        public decimal Amount { get; set; }
    }
}
