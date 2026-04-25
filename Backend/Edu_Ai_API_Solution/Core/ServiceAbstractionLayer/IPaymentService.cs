using Shared.Dtos.PaymentDto;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
    public interface IPaymentService
    {
        Task<string> CreatePaymentAsync(string studentId );

        Task<WebhookResultDto> HandleWebhookAsync(string body, string sentHmac);
    }

}
