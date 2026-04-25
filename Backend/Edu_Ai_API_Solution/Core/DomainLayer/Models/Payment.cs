using DomainLayer.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
    public class Payment:BaseEntity<int>
    {
        public string StudentId { get; set; }

        
        public decimal Amount { get; set; }
        public string Currency { get; set; } = null!;
        public PaymentStatus Status { get; set; } = PaymentStatus.Pending;
        public string TransactionId { get; set; }
        public DateTime PaymentDate { get; set; }
        public PaymentGateway PaymentMethod { get; set; }
        public ApplicationUser Student { get; set; } = null!;
        public int AcademicYearId { get; set; }

        // Navigation properties
        //public AcademicYear AcademicYear { get; set; }
    }
}
