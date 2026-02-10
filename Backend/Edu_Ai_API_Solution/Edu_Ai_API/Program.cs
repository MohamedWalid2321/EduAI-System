using DomainLayer.Contracts;
using Microsoft.EntityFrameworkCore;
using persistenceLayer.Data;
using persistenceLayer.Repository;
using ServiceAbstractionLayer;
using ServiceLayer;
using persistenceLayer;
using Edu_Ai_API.CustomMiddleWares;
using Microsoft.AspNetCore.Mvc;
using Shared.ErrorModels;
using Edu_Ai_API.Factories;
using StackExchange.Redis;
using ServiceLayer.Services;

namespace Edu_Ai_API
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Add services to the container.

            builder.Services.AddControllers();
            // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();
            builder.Services.AddInfrastructureServices(builder.Configuration);
            builder.Services.AddApplicationServices();

            builder.Services.Configure<ApiBehaviorOptions>((options) =>
            { 
                options.InvalidModelStateResponseFactory = ApiResponseFactory.GenerateApiValidationResponse;
            });

           //builder.Services.AddScoped<ICasheRepository, CasheRepository>();



            var app = builder.Build();


            

            app.MapGet("/", async ([FromServices] ICacheService cache) =>
            {
                await cache.SetAsync("mykey", "Helloo", TimeSpan.FromMinutes(2));
                return await cache.GetAsync("mykey");
            });

            


            app.UseMiddleware<CustomExceptionHandlerMiddleWare>();

            // Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseHttpsRedirection();

            app.UseAuthorization();

            app.MapControllers();

            app.Run();
        }
    }
}
