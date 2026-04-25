using Shared.Dtos.UserDto.Response;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.PaymentDto
{
    public class CreatePaymentRequestDto
    {
        public decimal Amount { get; set; }
        public string Currency { get; set; } = "EGP";
        public StudentBillingDto Student { get; set; }
        public string OrderId { get; set; }

         
    }
}
