using Shared.Dtos.PaymentDto;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
    public interface IPaymobService
    {
        Task<string> CreatePaymentUrlAsync(CreatePaymentRequestDto request);
        string CalculateHmac(WebhookObj obj);
    }
}
