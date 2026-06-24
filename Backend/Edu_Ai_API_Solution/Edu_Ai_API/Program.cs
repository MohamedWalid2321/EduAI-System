using DomainLayer.Models;
using Edu_Ai_API.CustomMiddleWares;
using Edu_Ai_API.Factories;
using Hangfire;
using HangfireBasicAuthenticationFilter;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.OpenApi.Models;
using persistenceLayer;
using Microsoft.EntityFrameworkCore;
using persistenceLayer.Data;
using PresentationLayer.Authorization;
using Serilog;
using ServiceAbstractionLayer;
using ServiceLayer;
using ServiceLayer.Jobs;

namespace Edu_Ai_API
{
    public class Program
    {
        public static async Task Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Configure Serilog
            builder.Host.UseSerilog((context, configuration) =>
                configuration.ReadFrom.Configuration(context.Configuration));

            builder.Services.AddControllers();
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen(options =>
            {
                options.SwaggerDoc("v1", new OpenApiInfo
                {
                    Title = "Lumino API",
                    Version = "v1",
                    Description = "Learning Management System API",
                    Contact = new OpenApiContact
                    {
                        Name = "Lumina Team",
                        Email = "support@lumino.com"
                    }
                });

                options.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
                {
                    Name = "Authorization",
                    Type = SecuritySchemeType.Http,
                    Scheme = "Bearer",
                    BearerFormat = "JWT",
                    In = ParameterLocation.Header,
                    Description = "Enter your JWT token"
                });

                options.AddSecurityRequirement(new OpenApiSecurityRequirement
                {
                    {
                        new OpenApiSecurityScheme
                        {
                            Reference = new OpenApiReference
                            {
                                Type = ReferenceType.SecurityScheme,
                                Id = "Bearer"
                            }
                        },
                        Array.Empty<string>()
                    }
                });
            });

            builder.Services.AddIdentity<ApplicationUser, ApplicationRole>()
                .AddEntityFrameworkStores<LmsDBContext>()
                .AddDefaultTokenProviders();

            builder.Services.AddInfrastructureServices(builder.Configuration);
            builder.Services.AddApplicationServices(builder.Environment, builder.Configuration);

            // Authorization
            builder.Services.AddSingleton<IAuthorizationPolicyProvider, PermissionAuthorizationPolicyProvider>();
            builder.Services.AddScoped<IAuthorizationHandler, PermissionAuthorizationHandler>();

            builder.Services.Configure<ApiBehaviorOptions>(options =>
            {
                options.InvalidModelStateResponseFactory = ApiResponseFactory.GenerateApiValidationResponse;
            });

            builder.Services.AddCors(options =>
            {
                options.AddPolicy("AllowAngular", corsBuilder =>
                {
                    corsBuilder.WithOrigins(
                        "http://localhost:4200",
                        "https://localhost:4200",
                        "http://localhost:4201",
                        "https://localhost:4201"
                    )
                    .AllowAnyMethod()
                    .AllowAnyHeader()
                    .AllowCredentials();
                });
            });

            builder.Services.Configure<FormOptions>(options =>
            {
                options.MultipartBodyLengthLimit = 524288000; // 500 MB
                options.ValueLengthLimit = int.MaxValue;
                options.MultipartHeadersLengthLimit = int.MaxValue;
            });

            var app = builder.Build();

            // Auto-apply any pending EF Core migrations on startup.
            // This keeps the production (cloud) database in sync automatically
            // every time a new version is published — no manual SQL scripts needed.
            using (var scope = app.Services.CreateScope())
            {
                var db = scope.ServiceProvider.GetRequiredService<LmsDBContext>();
                await db.Database.MigrateAsync();
            }

            // CORS must be first to handle preflight requests
            app.UseCors("AllowAngular");
            app.UseSerilogRequestLogging();
            app.UseMiddleware<CustomExceptionHandlerMiddleWare>();

            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI(c =>
                {
                    c.SwaggerEndpoint("/swagger/v1/swagger.json", "Lumino API V1");
                });

                app.UseHangfireDashboard("/jobs", new DashboardOptions
                {
                    Authorization =
                    [
                        new HangfireCustomBasicAuthenticationFilter
                        {
                            User = app.Configuration.GetValue<string>("HangfireSettings:Username"),
                            Pass = app.Configuration.GetValue<string>("HangfireSettings:Password")
                        }
                    ],
                    DashboardTitle = "Lumina Dashboard",
                });
            }

            // Schedule recurring background job: runs every hour
            var recurringJobManager = app.Services.GetRequiredService<IRecurringJobManager>();
            recurringJobManager.AddOrUpdate<AssignmentDeadlineReminderJob>(
                "assignment-deadline-reminder",
                job => job.SendDeadlineRemindersAsync(),
                Cron.Daily);

            app.UseHttpsRedirection();
            app.UseAuthentication();
            app.UseAuthorization();
            app.MapControllers();

            app.Run();
        }
    }
}
