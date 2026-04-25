using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
    public class BillingData
    {
        public string FirstName { get; set; } = null!;
        public string LastName { get; set; } = null!;
        public string Email { get; set; } = null!;
        public string PhoneNumber { get; set; } = null!;
        public string City { get; set; } = "Cairo";
        public string Country { get; set; } = "EG";
    }
}
