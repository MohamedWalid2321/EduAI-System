using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.PaymentSpecifications
{
    public class PaymentByOrderIdSpec : BaseSpecification<Payment, int>
    {
        public PaymentByOrderIdSpec(string orderId)
            : base(p => p.Id.ToString() == orderId)
        {
        }
    }
}
