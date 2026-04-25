using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.PaymentDto
{
    public class OrderResponse
    {
        public int Id { get; set; }

        public int AmountCents { get; set; }

        public string Currency { get; set; }

        public DateTime CreatedAt { get; set; }
    }
}
