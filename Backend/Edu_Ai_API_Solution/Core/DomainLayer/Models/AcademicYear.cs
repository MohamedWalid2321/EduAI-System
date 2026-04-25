using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
    public class AcademicYear :BaseEntity<int>
    {
        
        public string Name { get; set; }

        public ICollection<Fee> Fees { get; set; }
        public ICollection<Payment> Payments { get; set; }
    }
}
