using DomainLayer.Enums;
using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using ServiceAbstractionLayer;
using Shared.Dtos.PaymentDto;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace PresentationLayer.Controllers
{
    public class PaymentController(IServiceManager _serviceManager) : ApiControllerBase
    {
        [HttpPost("Start")]
        public async Task<IActionResult> Start()
        {
            var userId = User.GetUserId();
            var student = await _serviceManager.UserService.GetAsync(userId);

            if (student == null) return NotFound("Student not found");

            var url = await _serviceManager.PaymentService.CreatePaymentAsync(userId);

            return Ok(new { paymentUrl = url });
        }

        [HttpPost("webhook")]
        public async Task<IActionResult> Webhook()
        {
            try
            {
                using var reader = new StreamReader(Request.Body);
                var body = await reader.ReadToEndAsync();

                var sentHmac = Request.Query["hmac"].ToString();

                var result = await _serviceManager.PaymentService.HandleWebhookAsync(body , sentHmac);

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
