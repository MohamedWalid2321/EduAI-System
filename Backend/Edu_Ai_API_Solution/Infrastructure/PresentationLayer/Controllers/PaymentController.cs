using DomainLayer.Enums;
using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using ServiceAbstractionLayer;
using Shared.Dtos.PaymentDto;
using System.Diagnostics;
using System.Text.Json;

namespace PresentationLayer.Controllers
{
    public class PaymentController(IServiceManager _serviceManager) : ApiControllerBase
    {
        [HttpPost("Start")]
        public async Task<IActionResult> Start(CancellationToken cancellationToken)
        {
            var userId = User.GetUserId();
            var student = await _serviceManager.UserService.GetAsync(userId, cancellationToken);

            if (student == null) return NotFound("Student not found");

            var url = await _serviceManager.PaymentService.CreatePaymentAsync(userId, cancellationToken);

            return Ok(new { paymentUrl = url });
        }

        [HttpPost("webhook")]
        public async Task<IActionResult> Webhook(CancellationToken cancellationToken)
        {
            try
            {
                using var reader = new StreamReader(Request.Body);
                var body = await reader.ReadToEndAsync(cancellationToken);

                var sentHmac = Request.Query["hmac"].ToString();

                var result = await _serviceManager.PaymentService.HandleWebhookAsync(body, sentHmac, cancellationToken);

                return Ok(result);
            }
            catch (Exception ex)
            {
                Debug.WriteLine("WEBHOOK ERROR: " + ex.Message);
                Debug.WriteLine(ex.StackTrace);

                return StatusCode(500, ex.Message);
            }
        }
    }
}
